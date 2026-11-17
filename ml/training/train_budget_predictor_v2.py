"""
STS Fintech – Milestone 2 – Step 7.3B-3
Model D: V2 Budget Utilization Prediction & Overrun Forecasting Engine

Trains an enterprise budget predictor using snapshot ledgers (Day 7, 14, 21),
evaluates against linear run-rate and historical median baselines,
strictly excludes downstream leakage columns, and preserves all existing artifacts.
"""

import os
import calendar
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = "data/v2/raw"
OUTPUT_DIR = "ml/models/v2"
MODEL_PATH = os.path.join(OUTPUT_DIR, "budget_predictor_v2.joblib")


def load_and_preprocess_budget_data(data_dir: str = DATA_DIR):
    """
    Loads budget snapshots, budgets ledger, and department profiles,
    and constructs chronological features strictly without target leakage.
    """
    snapshots_path = os.path.join(data_dir, "budget_snapshots.csv")
    budgets_path = os.path.join(data_dir, "budgets.csv")
    depts_path = os.path.join(data_dir, "departments.csv")

    if not os.path.exists(snapshots_path) or not os.path.exists(depts_path):
        raise FileNotFoundError(f"Missing required data files in {data_dir}")

    snapshots = pd.read_csv(snapshots_path)
    departments = pd.read_csv(depts_path)
    budgets = pd.read_csv(budgets_path) if os.path.exists(budgets_path) else None

    # Merge department reference info
    snapshots = snapshots.merge(
        departments[["department_id", "department_code", "annual_budget_baseline", "division"]],
        on="department_id",
        how="left"
    )

    # Feature Engineering (strictly available at snapshot timestamp)
    snapshots["days_in_month"] = snapshots.apply(
        lambda r: calendar.monthrange(int(r["budget_year"]), int(r["budget_month"]))[1],
        axis=1
    )
    snapshots["remaining_days"] = snapshots["days_in_month"] - snapshots["as_of_day"]
    snapshots["linear_runrate"] = snapshots["month_to_date_actual"] / np.maximum(1e-4, (snapshots["elapsed_month_pct"] / 100.0))
    snapshots["mtd_utilization_pct"] = (snapshots["month_to_date_actual"] / np.maximum(1.0, snapshots["allocated_amount"])) * 100.0
    snapshots["linear_runrate_utilization_pct"] = (snapshots["linear_runrate"] / np.maximum(1.0, snapshots["allocated_amount"])) * 100.0
    snapshots["tx_rate_per_day"] = snapshots["transaction_count_mtd"] / np.maximum(1, snapshots["as_of_day"])
    snapshots["avg_tx_size_mtd"] = snapshots["month_to_date_actual"] / np.maximum(1, snapshots["transaction_count_mtd"])
    snapshots["alloc_vs_baseline"] = snapshots["allocated_amount"] / np.maximum(1.0, (snapshots["annual_budget_baseline"] / 12.0))
    snapshots["remaining_budget_mtd"] = snapshots["allocated_amount"] - snapshots["month_to_date_actual"]

    # Historical department profiles computed up to 2026-08 (no test set leakage)
    train_val_mask = (snapshots["budget_year"] <= 2025) | (
        (snapshots["budget_year"] == 2026) & (snapshots["budget_month"] <= 8)
    )
    hist_subset = snapshots[train_val_mask]
    dept_profiles = {}
    for did, group in hist_subset.groupby("department_id"):
        dept_profiles[int(did)] = {
            "hist_mean_util": float(group["final_utilization_pct"].mean()),
            "hist_median_util": float(group["final_utilization_pct"].median()),
            "hist_overrun_freq": float(group["is_overspent_final"].mean()),
            "annual_budget_baseline": float(group["annual_budget_baseline"].iloc[0]),
            "division": str(group["division"].iloc[0])
        }

    return snapshots, dept_profiles


FEATURE_COLS = [
    "department_id",
    "budget_month",
    "as_of_day",
    "elapsed_month_pct",
    "allocated_amount",
    "month_to_date_actual",
    "transaction_count_mtd",
    "days_in_month",
    "remaining_days",
    "linear_runrate",
    "mtd_utilization_pct",
    "linear_runrate_utilization_pct",
    "tx_rate_per_day",
    "avg_tx_size_mtd",
    "alloc_vs_baseline",
    "remaining_budget_mtd"
]


def train_budget_predictor_v2(data_dir: str = DATA_DIR, output_path: str = MODEL_PATH):
    """
    Trains and evaluates the V2 Budget Utilization Predictor.
    """
    print("=" * 70)
    print("STS FINTECH – STEP 7.3B-3: MODEL D – BUDGET UTILIZATION PREDICTOR")
    print("=" * 70)

    snapshots, dept_profiles = load_and_preprocess_budget_data(data_dir)

    # Chronological Split
    train_mask = (snapshots["budget_year"] <= 2025)
    val_mask = (snapshots["budget_year"] == 2026) & (snapshots["budget_month"] <= 8)
    test_mask = (snapshots["budget_year"] == 2026) & (snapshots["budget_month"].isin([9, 10]))
    stress_mask = (snapshots["budget_year"] == 2026) & (snapshots["budget_month"].isin([11, 12]))

    train_val_mask = train_mask | val_mask

    print(f"Dataset summary:")
    print(f"  Training samples (2024–2025):        {train_mask.sum():,}")
    print(f"  Validation samples (Jan–Aug 2026):   {val_mask.sum():,}")
    print(f"  Test samples (Sep–Oct 2026):         {test_mask.sum():,}")
    print(f"  Stress test samples (Nov–Dec 2026):  {stress_mask.sum():,}")

    X_train = snapshots.loc[train_val_mask, FEATURE_COLS]
    y_train = snapshots.loc[train_val_mask, "final_utilization_pct"]

    X_test = snapshots.loc[test_mask, FEATURE_COLS]
    y_test = snapshots.loc[test_mask, "final_utilization_pct"]
    y_act_test = snapshots.loc[test_mask, "final_actual_amount"]

    X_stress = snapshots.loc[stress_mask, FEATURE_COLS]
    y_stress = snapshots.loc[stress_mask, "final_utilization_pct"]
    y_act_stress = snapshots.loc[stress_mask, "final_actual_amount"]

    # 1. Baseline Comparisons on Test Set
    # Baseline A: Department Historical Median
    pred_base_median = X_test["department_id"].map(
        lambda d: dept_profiles.get(d, {}).get("hist_median_util", 85.0)
    )
    mae_base_median = mean_absolute_error(y_test, pred_base_median)

    # Baseline B: Linear Run-rate (elapsed_month_pct extrapolation)
    pred_base_linear = X_test["linear_runrate_utilization_pct"]
    mae_base_linear = mean_absolute_error(y_test, pred_base_linear)
    pred_base_linear_act = X_test["linear_runrate"]
    mae_base_linear_act = mean_absolute_error(y_act_test, pred_base_linear_act)

    print("\nBaseline Evaluation on Test Set (Sep–Oct 2026):")
    print(f"  Historical Median Utilization MAE: {mae_base_median:.2f}%")
    print(f"  Linear Run-rate Utilization MAE:   {mae_base_linear:.2f}%")
    print(f"  Linear Run-rate Amount MAE:        ₹{mae_base_linear_act:,.2f}")

    # 2. Train Candidates: Ridge, HistGBM, RandomForest
    ridge = Ridge(alpha=100.0)
    ridge.fit(X_train, y_train)
    mae_ridge = mean_absolute_error(y_test, ridge.predict(X_test))

    hgbm = HistGradientBoostingRegressor(random_state=42, max_iter=200, min_samples_leaf=15)
    hgbm.fit(X_train, y_train)
    mae_hgbm = mean_absolute_error(y_test, hgbm.predict(X_test))

    rf = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    pred_rf_act = X_test["allocated_amount"] * pred_rf / 100.0
    mae_rf = mean_absolute_error(y_test, pred_rf)
    r2_rf = r2_score(y_test, pred_rf)
    mae_rf_act = mean_absolute_error(y_act_test, pred_rf_act)
    r2_rf_act = r2_score(y_act_test, pred_rf_act)

    print("\nModel Candidate Comparison on Test Set (Sep–Oct 2026):")
    print(f"  Ridge Regressor Util MAE:          {mae_ridge:.2f}%")
    print(f"  HistGBM Regressor Util MAE:        {mae_hgbm:.2f}%")
    print(f"  RandomForest Regressor Util MAE:   {mae_rf:.2f}% (R²: {r2_rf:.4f})")
    print(f"  RandomForest Regressor Amount MAE: ₹{mae_rf_act:,.2f} (R²: {r2_rf_act:.4f})")

    # Selected Model: RandomForestRegressor
    selected_model = rf

    # Snapshot Breakdown (Day 7, 14, 21)
    day_metrics = {}
    print("\nPerformance by Snapshot Day (Test Set Sep–Oct 2026):")
    for day in [7, 14, 21]:
        d_mask = X_test["as_of_day"] == day
        d_y_true = y_test[d_mask]
        d_y_pred = pred_rf[d_mask]
        d_act_true = y_act_test[d_mask]
        d_act_pred = pred_rf_act[d_mask]
        d_base_util = pred_base_linear[d_mask]

        d_mae_util = mean_absolute_error(d_y_true, d_y_pred)
        d_r2_util = r2_score(d_y_true, d_y_pred)
        d_mae_act = mean_absolute_error(d_act_true, d_act_pred)
        d_r2_act = r2_score(d_act_true, d_act_pred)
        d_base_mae = mean_absolute_error(d_y_true, d_base_util)

        day_metrics[f"day_{day}"] = {
            "sample_count": int(d_mask.sum()),
            "baseline_linear_util_mae": round(float(d_base_mae), 2),
            "model_util_mae": round(float(d_mae_util), 2),
            "model_util_r2": round(float(d_r2_util), 4),
            "model_amount_mae": round(float(d_mae_act), 2),
            "model_amount_r2": round(float(d_r2_act), 4)
        }
        print(f"  Day {day:02d} Snapshot ({d_mask.sum()} samples):")
        print(f"    Baseline Linear Util MAE: {d_base_mae:.2f}%")
        print(f"    Model Util MAE:           {d_mae_util:.2f}% (R²: {d_r2_util:.4f})")
        print(f"    Model Amount MAE:         ₹{d_mae_act:,.2f} (R²: {d_r2_act:.4f})")

    # Residual standard deviation for overrun confidence calculation
    residuals = y_test - pred_rf
    residual_std = float(np.std(residuals))

    # Synthetic Future Stress Test Evaluation (Nov–Dec 2026)
    pred_stress = rf.predict(X_stress)
    pred_stress_act = X_stress["allocated_amount"] * pred_stress / 100.0
    mae_stress_util = mean_absolute_error(y_stress, pred_stress)
    mae_stress_act = mean_absolute_error(y_act_stress, pred_stress_act)

    print("\nSynthetic Future Stress Test (Nov–Dec 2026, strictly isolated):")
    print(f"  Stress Util MAE:   {mae_stress_util:.2f}%")
    print(f"  Stress Amount MAE: ₹{mae_stress_act:,.2f}")

    # Save Bundle
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bundle = {
        "model": selected_model,
        "model_name": "RandomForestRegressor",
        "feature_names": FEATURE_COLS,
        "department_profiles": dept_profiles,
        "residual_std": residual_std,
        "overall_test_metrics": {
            "utilization_mae": round(float(mae_rf), 2),
            "utilization_rmse": round(float(np.sqrt(mean_squared_error(y_test, pred_rf))), 2),
            "utilization_r2": round(float(r2_rf), 4),
            "amount_mae": round(float(mae_rf_act), 2),
            "amount_r2": round(float(r2_rf_act), 4),
            "test_sample_count": len(X_test)
        },
        "day_metrics": day_metrics,
        "baselines": {
            "historical_median_util_mae": round(float(mae_base_median), 2),
            "linear_runrate_util_mae": round(float(mae_base_linear), 2),
            "linear_runrate_amount_mae": round(float(mae_base_linear_act), 2)
        },
        "stress_test_metrics": {
            "utilization_mae": round(float(mae_stress_util), 2),
            "amount_mae": round(float(mae_stress_act), 2),
            "sample_count": len(X_stress)
        }
    }

    joblib.dump(bundle, output_path)
    print(f"\nArtifact saved to: {output_path} ({os.path.getsize(output_path):,} bytes)")
    return bundle


if __name__ == "__main__":
    train_budget_predictor_v2()
