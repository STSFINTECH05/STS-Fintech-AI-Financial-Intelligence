"""
STS FINTECH AI FINANCIAL INTELLIGENCE
Model C: V2 Expense Forecasting Training Pipeline (Milestone 2 - Step 7.3B-2)

Builds enterprise-level continuous daily expenditure time-series forecasting.
Enforces strict chronological splitting to eliminate look-ahead bias:
- Training:               2024-01-31 to 2025-12-31 (701 days after 30-day lag buffer)
- Validation:             2026-01-01 to 2026-08-31 (243 days) - Strictly for model selection
- Historical Holdout Test: 2026-09-01 to 2026-10-31 (61 days)
- Synthetic Future Stress: 2026-11-14 to 2026-12-31 (48 days) - Partitioned separately

Candidate Models Evaluated:
1. Seasonal Naive baseline (7-day seasonal lag)
2. Previous-Day Naive baseline (1-day lag)
3. Historical Moving Average baseline (7-day rolling mean)
4. Ridge Regression (L2 regularized)
5. HistGradientBoostingRegressor (Gradient Boosted Trees)
6. RandomForestRegressor

Persists optimal candidate to ml/models/v2/expense_forecaster_v2.joblib.
"""

import os
import sys
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

DATA_PATH = os.path.join(BASE_DIR, "data", "v2", "raw", "transactions.csv")
V2_MODELS_DIR = os.path.join(BASE_DIR, "ml", "models", "v2")
os.makedirs(V2_MODELS_DIR, exist_ok=True)

# Indian public holiday calendar dates modeled in data generator
INDIAN_HOLIDAYS = {(1, 26), (8, 15), (10, 2), (11, 1), (12, 25)}


def load_and_aggregate_v2_daily_expenses(
    csv_path: str = DATA_PATH
) -> pd.DataFrame:
    """
    Aggregates V2 transaction records into a continuous daily expenditure time series.
    Verifies 100% mathematical reconciliation with raw transactions.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"V2 Transactions CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    df["parsed_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    if df["parsed_date"].isna().any():
        raise ValueError("Failed to parse some transaction dates in transactions.csv")

    raw_sum = float(df["amount"].sum())

    daily = df.groupby(df["parsed_date"].dt.date).agg(
        daily_expenditure=("amount", "sum"),
        transaction_count=("amount", "count"),
        average_amount=("amount", "mean"),
        median_amount=("amount", "median"),
        max_amount=("amount", "max")
    ).reset_index()

    daily["date"] = pd.to_datetime(daily["parsed_date"])
    daily = daily.drop(columns=["parsed_date"]).sort_values("date").reset_index(drop=True)

    # Re-index to complete calendar day continuum
    min_date = daily["date"].min()
    max_date = daily["date"].max()
    full_idx = pd.date_range(min_date, max_date, freq="D")

    daily = daily.set_index("date").reindex(full_idx)
    daily["daily_expenditure"] = daily["daily_expenditure"].fillna(0.0)
    daily["transaction_count"] = daily["transaction_count"].fillna(0).astype(int)
    daily["average_amount"] = daily["average_amount"].fillna(0.0)
    daily["median_amount"] = daily["median_amount"].fillna(0.0)
    daily["max_amount"] = daily["max_amount"].fillna(0.0)
    daily = daily.reset_index().rename(columns={"index": "date"})

    agg_sum = float(daily["daily_expenditure"].sum())
    if not np.isclose(raw_sum, agg_sum, rtol=1e-5):
        raise ValueError(f"Reconciliation mismatch! Raw sum: {raw_sum}, Aggregated sum: {agg_sum}")

    return daily


def extract_v2_forecasting_features(
    daily_df: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extracts calendar, shifted lag, and shifted rolling window features.
    ZERO LEAKAGE GUARD: Target day expenditure is strictly shifted so it is NEVER
    incorporated into feature calculations for that day.
    """
    df = daily_df.copy()

    # 1. Calendar Features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_end"] = (df["day_of_month"] >= 25).astype(int)
    df["is_month_start"] = (df["day_of_month"] <= 3).astype(int)
    df["is_holiday"] = df["date"].apply(lambda dt: int((dt.month, dt.day) in INDIAN_HOLIDAYS))

    # 2. Shifted Lag features (t - l)
    for l in [1, 2, 3, 7, 14, 21, 28, 30]:
        df[f"lag_{l}"] = df["daily_expenditure"].shift(l)

    # 3. Shifted Rolling statistics (strictly shifted by 1 to prevent current-day leakage)
    shifted = df["daily_expenditure"].shift(1)
    df["rolling_mean_7"] = shifted.rolling(window=7, min_periods=1).mean()
    df["rolling_std_7"] = shifted.rolling(window=7, min_periods=1).std().fillna(0.0)
    df["rolling_mean_14"] = shifted.rolling(window=14, min_periods=1).mean()
    df["rolling_mean_30"] = shifted.rolling(window=30, min_periods=1).mean()
    df["rolling_std_30"] = shifted.rolling(window=30, min_periods=1).std().fillna(0.0)
    df["rolling_min_7"] = shifted.rolling(window=7, min_periods=1).min()
    df["rolling_max_7"] = shifted.rolling(window=7, min_periods=1).max()
    df["lag1_to_roll7"] = df["lag_1"] / np.maximum(df["rolling_mean_7"], 1.0)
    df["roll7_to_roll30"] = df["rolling_mean_7"] / np.maximum(df["rolling_mean_30"], 1.0)

    feature_cols = [
        "day_of_week",
        "day_of_month",
        "month",
        "quarter",
        "day_of_year",
        "is_weekend",
        "is_month_end",
        "is_month_start",
        "is_holiday",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_7",
        "lag_14",
        "lag_21",
        "lag_28",
        "lag_30",
        "rolling_mean_7",
        "rolling_std_7",
        "rolling_mean_14",
        "rolling_mean_30",
        "rolling_std_30",
        "rolling_min_7",
        "rolling_max_7",
        "lag1_to_roll7",
        "roll7_to_roll30"
    ]

    return df, feature_cols


def train_expense_forecaster_v2(
    random_state: int = 42
) -> Tuple[Any, Dict[str, Any]]:
    """
    Trains, compares candidate models, selects optimal forecaster on validation set,
    and serializes ml/models/v2/expense_forecaster_v2.joblib.
    """
    print("=" * 70)
    print("STS FINTECH - MODEL C: V2 EXPENSE FORECASTER TRAINING")
    print("=" * 70)

    # 1. Ingestion & Aggregation
    daily_df = load_and_aggregate_v2_daily_expenses()
    total_days = len(daily_df)
    total_spend = float(daily_df["daily_expenditure"].sum())

    print(f"Total Calendar Days: {total_days} ({daily_df['date'].min().date()} to {daily_df['date'].max().date()})")
    print(f"Total Expenditure:   INR {total_spend:,.2f}")

    # 2. Feature Extraction
    feat_df, feature_cols = extract_v2_forecasting_features(daily_df)
    # Drop first 30 days due to 30-day lag buffer
    valid_data = feat_df.dropna().reset_index(drop=True)

    # 3. Chronological Partitions
    train_mask = (valid_data["date"] >= "2024-01-31") & (valid_data["date"] <= "2025-12-31")
    val_mask = (valid_data["date"] >= "2026-01-01") & (valid_data["date"] <= "2026-08-31")
    test_mask = (valid_data["date"] >= "2026-09-01") & (valid_data["date"] <= "2026-10-31")
    stress_mask = (valid_data["date"] >= "2026-11-14") & (valid_data["date"] <= "2026-12-31")

    X_train, y_train = valid_data.loc[train_mask, feature_cols], valid_data.loc[train_mask, "daily_expenditure"]
    X_val, y_val = valid_data.loc[val_mask, feature_cols], valid_data.loc[val_mask, "daily_expenditure"]
    X_test, y_test = valid_data.loc[test_mask, feature_cols], valid_data.loc[test_mask, "daily_expenditure"]

    print("\n--- Chronological Split Breakdown ---")
    print(f"  Training (2024-01-31 to 2025-12-31):        {len(X_train)} days")
    print(f"  Validation (2026-01-01 to 2026-08-31):      {len(X_val)} days (used strictly for model selection)")
    print(f"  Historical Holdout (2026-09-01 to 2026-10-31): {len(X_test)} days")
    print(f"  Synthetic Future Stress (Nov 14 to Dec 31):   {stress_mask.sum()} days")

    # 4. Evaluate Baselines on Validation Set
    val_actual_sum = float(y_val.sum())
    baselines_eval = {}

    def compute_metrics(y_true, y_pred, model_name: str) -> Dict[str, float]:
        y_pred = np.maximum(y_pred, 0.0)
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        wape = float(np.sum(np.abs(y_true - y_pred)) / np.sum(y_true))
        bias = float(np.sum(y_pred - y_true) / np.sum(y_true))
        mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1.0))))
        print(f"  {model_name:32s} | MAE: INR {mae:10,.2f} | RMSE: INR {rmse:10,.2f} | WAPE: {wape*100:6.2f}% | Bias: {bias*100:6.2f}%")
        return {"mae": mae, "rmse": rmse, "wape": wape, "bias": bias, "mape": mape}

    print("\n--- Baseline Models Evaluation on Validation Set ---")
    baselines_eval["Seasonal_Naive_lag_7"] = compute_metrics(y_val, valid_data.loc[val_mask, "lag_7"], "Seasonal Naive (lag_7)")
    baselines_eval["Previous_Day_Naive_lag_1"] = compute_metrics(y_val, valid_data.loc[val_mask, "lag_1"], "1-Day Naive (lag_1)")
    baselines_eval["Rolling_7Day_Mean"] = compute_metrics(y_val, valid_data.loc[val_mask, "rolling_mean_7"], "Rolling 7-Day Mean")

    # 5. Candidate Models Comparison on Validation Set
    candidates = {
        "Ridge_Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=100.0, random_state=random_state))
        ]),
        "HistGradientBoosting_L1": HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=200,
            min_samples_leaf=15,
            random_state=random_state
        ),
        "HistGradientBoosting_L2": HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=200,
            min_samples_leaf=15,
            random_state=random_state
        ),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )
    }

    print("\n--- ML Candidate Models Evaluation on Validation Set ---")
    candidate_val_results = {}
    best_candidate_name = None
    best_candidate_model = None
    best_val_wape = float("inf")

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        res = compute_metrics(y_val, val_preds, name)
        candidate_val_results[name] = res

        if res["wape"] < best_val_wape:
            best_val_wape = res["wape"]
            best_candidate_name = name
            best_candidate_model = model

    print(f"\nOptimal Model Selected on Validation: {best_candidate_name} (Validation WAPE: {best_val_wape*100:.2f}%)")

    # 6. Fit Selected Model on Complete Available Pre-Test History (up to 2026-08-31)
    full_train_mask = (valid_data["date"] >= "2024-01-31") & (valid_data["date"] <= "2026-08-31")
    X_full_train = valid_data.loc[full_train_mask, feature_cols]
    y_full_train = valid_data.loc[full_train_mask, "daily_expenditure"]

    best_candidate_model.fit(X_full_train, y_full_train)

    # 7. Single-Step Historical Test Evaluation
    test_preds = best_candidate_model.predict(X_test)
    test_metrics = compute_metrics(y_test, test_preds, f"Final Test ({best_candidate_name})")

    # Calculate validation residuals for empirical prediction interval calibration
    val_preds_fitted = best_candidate_model.predict(X_val)
    val_residuals = np.array(y_val) - np.array(val_preds_fitted)
    residual_std = float(np.std(val_residuals))
    residual_q80 = float(np.percentile(np.abs(val_residuals), 80))
    residual_q95 = float(np.percentile(np.abs(val_residuals), 95))

    # 8. Save Artifact to ml/models/v2/
    model_artifact_path = os.path.join(V2_MODELS_DIR, "expense_forecaster_v2.joblib")
    joblib.dump(best_candidate_model, model_artifact_path)

    with open(model_artifact_path, "rb") as f:
        artifact_hash = hashlib.sha256(f.read()).hexdigest()

    print(f"\nSaved V2 Expense Forecaster to: {model_artifact_path}")
    print(f"Artifact SHA-256: {artifact_hash}")

    metadata = {
        "model_version": "2.0.0",
        "selected_model": best_candidate_name,
        "artifact_path": model_artifact_path,
        "sha256": artifact_hash,
        "training_timestamp": datetime.now().isoformat(),
        "training_date_range": ["2024-01-31", "2026-08-31"],
        "historical_test_date_range": ["2026-09-01", "2026-10-31"],
        "synthetic_future_stress_range": ["2026-11-14", "2026-12-31"],
        "feature_columns": feature_cols,
        "baselines_validation": baselines_eval,
        "candidate_validation": candidate_val_results,
        "historical_test_metrics": test_metrics,
        "prediction_intervals": {
            "residual_std": residual_std,
            "error_margin_80_percent": residual_q80,
            "error_margin_95_percent": residual_q95
        }
    }

    return best_candidate_model, metadata


if __name__ == "__main__":
    train_expense_forecaster_v2()
