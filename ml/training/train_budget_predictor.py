"""
STS Fintech – Milestone 2 – Step 7.2D: Budget Prediction Training Pipeline
Trains a machine learning model to predict departmental budget utilization and overspending risk.
Adheres strictly to chronological splitting, target leakage prevention, and baseline comparisons.
"""

import os
import json
import hashlib
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support, confusion_matrix


DATA_DIR = "data/raw"
BUDGETS_PATH = os.path.join(DATA_DIR, "budgets.csv")
TRANSACTIONS_PATH = os.path.join(DATA_DIR, "transactions.csv")
MODELS_DIR = "ml/models"
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "budget_predictor.joblib")


def calculate_sha256(filepath: str) -> str:
    """Calculate SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def audit_budget_transaction_reconciliation(budgets_df: pd.DataFrame, transactions_df: pd.DataFrame) -> dict:
    """
    Audits the financial relationship and reconciliation between budgets and transactions.
    Reports correlation, scale discrepancies, and independence findings.
    """
    tx_df = transactions_df.copy()
    tx_df["date"] = pd.to_datetime(tx_df["transaction_date"])
    tx_df["year"] = tx_df["date"].dt.year
    tx_df["month"] = tx_df["date"].dt.month

    # Aggregate budgets by department, year, month
    agg_b = budgets_df.groupby(["department_id", "budget_year", "budget_month"]).agg(
        allocated=("allocated_amount", "sum"),
        actual=("actual_amount", "sum"),
        line_count=("budget_id", "count")
    ).reset_index()

    # Aggregate transactions by department, year, month
    tx_agg = tx_df.groupby(["department_id", "year", "month"])["amount"].sum().reset_index()
    tx_agg.rename(columns={"amount": "tx_total"}, inplace=True)

    merged = pd.merge(
        agg_b,
        tx_agg,
        left_on=["department_id", "budget_year", "budget_month"],
        right_on=["department_id", "year", "month"],
        how="inner"
    )

    if len(merged) > 0:
        corr = float(merged["actual"].corr(merged["tx_total"]))
        ratio = float((merged["actual"] / merged["tx_total"].replace(0, np.nan)).median())
    else:
        corr = 0.0
        ratio = 0.0

    return {
        "matched_department_months": len(merged),
        "pearson_correlation": round(corr, 4),
        "median_budget_to_tx_ratio": round(ratio, 2),
        "reconciled": False,
        "limitation_note": (
            "Budgets actuals and transaction actuals originate from independent synthetic generators. "
            f"Correlation is {corr:.4f} with a ~{ratio:.1f}x scale difference. "
            "Transactions cannot serve as intra-month run-rate snapshots for macro budgets without inducing artificial noise."
        )
    }


def aggregate_monthly_budgets(budgets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates legitimate allocation lines by department_id, budget_year, and budget_month.
    Preserves all original allocation lines while computing monthly department totals.
    """
    agg = budgets_df.groupby(["department_id", "budget_year", "budget_month"]).agg(
        allocated_amount=("allocated_amount", "sum"),
        actual_amount=("actual_amount", "sum"),
        remaining_amount=("remaining_amount", "sum"),
        line_count=("budget_id", "count")
    ).reset_index()

    agg["utilization_percentage"] = (agg["actual_amount"] / agg["allocated_amount"]) * 100.0
    agg["period"] = agg["budget_year"].astype(str) + "-" + agg["budget_month"].astype(str).str.zfill(2)
    agg.sort_values(["department_id", "period"], inplace=True)
    agg.reset_index(drop=True, inplace=True)
    return agg


def engineer_budget_features(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs strictly leak-free historical departmental features.
    For department d at period t, uses strictly periods < t.
    """
    records = []
    for dept_id, group in agg_df.groupby("department_id"):
        group = group.copy().reset_index(drop=True)
        for i in range(len(group)):
            row = group.iloc[i].to_dict()
            prior = group.iloc[:i]
            if len(prior) == 0:
                row["hist_mean_util"] = np.nan
                row["hist_overrun_freq"] = np.nan
                row["hist_mean_alloc"] = np.nan
                row["hist_mean_actual"] = np.nan
                row["lag_1_util"] = np.nan
                row["lag_2_util"] = np.nan
                row["rolling_3m_util"] = np.nan
            else:
                row["hist_mean_util"] = float(prior["utilization_percentage"].mean())
                row["hist_overrun_freq"] = float((prior["utilization_percentage"] > 100.0).mean())
                row["hist_mean_alloc"] = float(prior["allocated_amount"].mean())
                row["hist_mean_actual"] = float(prior["actual_amount"].mean())
                row["lag_1_util"] = float(prior.iloc[-1]["utilization_percentage"])
                row["lag_2_util"] = float(prior.iloc[-2]["utilization_percentage"]) if len(prior) >= 2 else float(prior.iloc[-1]["utilization_percentage"])
                row["rolling_3m_util"] = float(prior.iloc[-3:]["utilization_percentage"].mean())
            records.append(row)

    feat_df = pd.DataFrame(records)
    # Drop first month warmup where historical mean is NaN
    feat_df = feat_df.dropna(subset=["hist_mean_util"]).reset_index(drop=True)

    # Calendar and log-transformed budget scale features
    feat_df["quarter"] = (feat_df["budget_month"] - 1) // 3 + 1
    feat_df["log_allocated"] = np.log1p(feat_df["allocated_amount"].clip(lower=0))
    feat_df["log_hist_mean_alloc"] = np.log1p(feat_df["hist_mean_alloc"].clip(lower=0))

    return feat_df


FEATURE_COLS = [
    "department_id",
    "budget_year",
    "budget_month",
    "quarter",
    "log_allocated",
    "line_count",
    "hist_mean_util",
    "hist_overrun_freq",
    "log_hist_mean_alloc",
    "lag_1_util",
    "lag_2_util",
    "rolling_3m_util"
]

EXCLUDED_LEAKAGE_COLS = [
    "actual_amount",
    "remaining_amount",
    "utilization_percentage",
    "is_overspent"
]


def classify_risk_tier(utilization: float) -> str:
    """Classifies utilization percentage into standardized business risk tiers."""
    if utilization < 80.0:
        return "Within Budget"
    elif utilization <= 100.0:
        return "Approaching Budget Limit"
    else:
        return "Potential Budget Overrun"


def train_budget_predictor():
    """Executes the complete budget prediction training and comparison pipeline."""
    print("=" * 70)
    print("STS FINTECH – STEP 7.2D: BUDGET PREDICTION TRAINING")
    print("=" * 70)

    # 1. Load data
    budgets_df = pd.read_csv(BUDGETS_PATH)
    transactions_df = pd.read_csv(TRANSACTIONS_PATH)

    print(f"Loaded {len(budgets_df):,} allocation-line records from budgets.csv")
    print(f"Loaded {len(transactions_df):,} transaction records from transactions.csv")

    # 2. Reconciliation audit
    reconciliation = audit_budget_transaction_reconciliation(budgets_df, transactions_df)
    print(f"Reconciliation: Pearson r = {reconciliation['pearson_correlation']}, Scale ratio = {reconciliation['median_budget_to_tx_ratio']}x")
    print(f"Note: {reconciliation['limitation_note']}")

    # 3. Monthly aggregation
    agg_df = aggregate_monthly_budgets(budgets_df)
    print(f"Aggregated {len(budgets_df):,} allocation lines into {len(agg_df):,} department-month records across 50 departments.")

    # 4. Feature engineering
    feat_df = engineer_budget_features(agg_df)
    print(f"Engineered {len(FEATURE_COLS)} features across {len(feat_df):,} department-month rows (warmup period dropped).")

    # 5. Chronological splitting
    train_mask = feat_df["period"] <= "2026-04"
    val_mask = (feat_df["period"] >= "2026-05") & (feat_df["period"] <= "2026-08")
    test_mask = feat_df["period"] >= "2026-09"

    X_train = feat_df.loc[train_mask, FEATURE_COLS].values.astype(np.float64)
    y_train = feat_df.loc[train_mask, "utilization_percentage"].values.astype(np.float64)

    X_val = feat_df.loc[val_mask, FEATURE_COLS].values.astype(np.float64)
    y_val = feat_df.loc[val_mask, "utilization_percentage"].values.astype(np.float64)

    X_test = feat_df.loc[test_mask, FEATURE_COLS].values.astype(np.float64)
    y_test = feat_df.loc[test_mask, "utilization_percentage"].values.astype(np.float64)

    print(f"Chronological split: Train={len(X_train)} (<= 2026-04), Val={len(X_val)} (2026-05 to 2026-08), Test={len(X_test)} (>= 2026-09)")

    # 6. Baseline & Model comparisons on Validation Set
    hist_mean_idx = FEATURE_COLS.index("hist_mean_util")
    y_pred_base_val = X_val[:, hist_mean_idx]
    base_mae = mean_absolute_error(y_val, y_pred_base_val)
    base_rmse = np.sqrt(mean_squared_error(y_val, y_pred_base_val))
    base_wape = (np.sum(np.abs(y_val - y_pred_base_val)) / np.sum(y_val)) * 100.0

    print("\n--- VALIDATION SET COMPARISON (195 department-months) ---")
    print(f"Baseline (Dept Hist Mean): MAE={base_mae:.2f}%, RMSE={base_rmse:.2f}%, WAPE={base_wape:.2f}%")

    # Candidate 1: Ridge Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    ridge = Ridge(alpha=50.0, solver="lsqr", random_state=42)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge_val = ridge.predict(X_val_scaled)
    ridge_mae = mean_absolute_error(y_val, y_pred_ridge_val)
    ridge_rmse = np.sqrt(mean_squared_error(y_val, y_pred_ridge_val))
    ridge_wape = (np.sum(np.abs(y_val - y_pred_ridge_val)) / np.sum(y_val)) * 100.0
    print(f"Ridge Regression (alpha=50): MAE={ridge_mae:.2f}%, RMSE={ridge_rmse:.2f}%, WAPE={ridge_wape:.2f}%")

    # Candidate 2: Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=20, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf_val = rf.predict(X_val)
    rf_mae = mean_absolute_error(y_val, y_pred_rf_val)
    rf_rmse = np.sqrt(mean_squared_error(y_val, y_pred_rf_val))
    rf_wape = (np.sum(np.abs(y_val - y_pred_rf_val)) / np.sum(y_val)) * 100.0
    print(f"Random Forest (trees=100): MAE={rf_mae:.2f}%, RMSE={rf_rmse:.2f}%, WAPE={rf_wape:.2f}%")

    # Candidate 3: HistGradientBoosting
    hgb = HistGradientBoostingRegressor(max_iter=50, max_depth=3, min_samples_leaf=25, l2_regularization=10.0, random_state=42)
    hgb.fit(X_train, y_train)
    y_pred_hgb_val = hgb.predict(X_val)
    hgb_mae = mean_absolute_error(y_val, y_pred_hgb_val)
    hgb_rmse = np.sqrt(mean_squared_error(y_val, y_pred_hgb_val))
    hgb_wape = (np.sum(np.abs(y_val - y_pred_hgb_val)) / np.sum(y_val)) * 100.0
    print(f"HistGradientBoosting: MAE={hgb_mae:.2f}%, RMSE={hgb_rmse:.2f}%, WAPE={hgb_wape:.2f}%")

    # Select model STRICTLY based on validation set performance (unbiased selection)
    val_performances = {
        "RidgeRegression": (ridge_mae, ridge, True),
        "RandomForest": (rf_mae, rf, False),
        "HistGradientBoosting": (hgb_mae, hgb, False)
    }
    # Unbiased selection using validation MAE only (never using test set)
    best_model_name = min(val_performances, key=lambda k: val_performances[k][0])
    best_mae, best_model, uses_scaler = val_performances[best_model_name]
    print(f"\nSelected Model: {best_model_name} (Validation MAE: {best_mae:.2f}%)")

    # 7. Final Evaluation on Held-Out Test Set (198 department-months)
    if uses_scaler:
        y_pred_test = best_model.predict(X_test_scaled)
    else:
        y_pred_test = best_model.predict(X_test)

    # Clamping predicted utilization to positive values
    y_pred_test = np.clip(y_pred_test, 0.0, 200.0)

    test_mae = float(mean_absolute_error(y_test, y_pred_test))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    test_wape = float((np.sum(np.abs(y_test - y_pred_test)) / np.sum(y_test)) * 100.0)

    # Baseline comparison on test set
    y_pred_base_test = X_test[:, hist_mean_idx]
    base_test_mae = float(mean_absolute_error(y_test, y_pred_base_test))
    base_test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_base_test)))
    base_test_wape = float((np.sum(np.abs(y_test - y_pred_base_test)) / np.sum(y_test)) * 100.0)

    # Overspending classification (> 100%)
    actual_overspent = (y_test > 100.0).astype(int)
    pred_overspent = (y_pred_test > 100.0).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(actual_overspent, pred_overspent, average="binary", zero_division=0)
    cm = confusion_matrix(actual_overspent, pred_overspent).tolist()

    # Risk tiers
    actual_tiers = [classify_risk_tier(u) for u in y_test]
    pred_tiers = [classify_risk_tier(u) for u in y_pred_test]

    print("\n--- HELD-OUT TEST SET EVALUATION (198 department-months: 2026-09 to 2026-12) ---")
    print(f"Test MAE: {test_mae:.2f}% (vs Baseline: {base_test_mae:.2f}%)")
    print(f"Test RMSE: {test_rmse:.2f}% (vs Baseline: {base_test_rmse:.2f}%)")
    print(f"Test WAPE: {test_wape:.2f}% (vs Baseline: {base_test_wape:.2f}%)")
    print(f"Overspending (>100%) Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"Confusion Matrix (Overspending [TN, FP], [FN, TP]): {cm}")

    # Build historical department profiles for inference lookup
    dept_profiles = {}
    for dept_id, group in feat_df.groupby("department_id"):
        latest = group.iloc[-1]
        dept_profiles[int(dept_id)] = {
            "hist_mean_util": float(latest["hist_mean_util"]),
            "hist_overrun_freq": float(latest["hist_overrun_freq"]),
            "hist_mean_alloc": float(latest["hist_mean_alloc"]),
            "lag_1_util": float(latest["lag_1_util"]),
            "lag_2_util": float(latest["lag_2_util"]),
            "rolling_3m_util": float(latest["rolling_3m_util"]),
            "typical_line_count": int(group["line_count"].median())
        }

    # 8. Bundle and save artifact
    os.makedirs(MODELS_DIR, exist_ok=True)
    bundle = {
        "model": best_model,
        "scaler": scaler if uses_scaler else None,
        "uses_scaler": uses_scaler,
        "model_name": best_model_name,
        "feature_names": FEATURE_COLS,
        "department_profiles": dept_profiles,
        "residual_std": float(np.std(y_train - (best_model.predict(X_train_scaled) if uses_scaler else best_model.predict(X_train))))
    }
    joblib.dump(bundle, MODEL_SAVE_PATH)
    checksum = calculate_sha256(MODEL_SAVE_PATH)
    print(f"\nSaved Budget Predictor artifact to {MODEL_SAVE_PATH} ({os.path.getsize(MODEL_SAVE_PATH):,} bytes)")
    print(f"SHA-256 Checksum: {checksum}")

    return {
        "model_name": best_model_name,
        "checksum": checksum,
        "reconciliation": reconciliation,
        "validation_metrics": {
            "Baseline": {"MAE": base_mae, "RMSE": base_rmse, "WAPE": base_wape},
            "RidgeRegression": {"MAE": ridge_mae, "RMSE": ridge_rmse, "WAPE": ridge_wape},
            "RandomForest": {"MAE": rf_mae, "RMSE": rf_rmse, "WAPE": rf_wape},
            "HistGradientBoosting": {"MAE": hgb_mae, "RMSE": hgb_rmse, "WAPE": hgb_wape}
        },
        "test_metrics": {
            "MAE": test_mae,
            "RMSE": test_rmse,
            "WAPE": test_wape,
            "Baseline_MAE": base_test_mae,
            "Baseline_RMSE": base_test_rmse,
            "Baseline_WAPE": base_test_wape,
            "overspending_precision": float(precision),
            "overspending_recall": float(recall),
            "overspending_f1": float(f1),
            "confusion_matrix": cm
        }
    }


if __name__ == "__main__":
    train_budget_predictor()
