"""
STS FINTECH AI FINANCIAL INTELLIGENCE
V2 Expense Forecaster Comprehensive Evaluation Pipeline (Milestone 2 - Step 7.3B-2)

Executes multi-origin rolling-origin chronological evaluation on:
- Historical Holdout Test period (2026-09-01 to 2026-10-31)
- Multi-horizon forecasts: 7-day, 14-day, and 30-day
- Computes Daily, Weekly, and Monthly metrics: MAE, RMSE, WAPE, MAPE, Bias
- Evaluates empirical prediction interval coverage
- Compares against 4 baselines: Seasonal Naive, 1-Day Naive, Moving Average, Ridge
- Evaluates Synthetic Future Stress Test (2026-11-14 onwards) separately
- Serializes ml/models/v2/expense_forecaster_v2_metadata.json with SHA-256 checksums.
"""

import os
import sys
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
import sklearn
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from ml.training.train_expense_forecaster_v2 import (
    load_and_aggregate_v2_daily_expenses,
    train_expense_forecaster_v2
)
from ml.prediction.expense_forecaster_v2 import ExpenseForecasterV2

V2_MODELS_DIR = os.path.join(BASE_DIR, "ml", "models", "v2")
METADATA_FILE = os.path.join(V2_MODELS_DIR, "expense_forecaster_v2_metadata.json")
GENERATOR_VERSION_FILE = os.path.join(BASE_DIR, "data", "v2", "GENERATOR_VERSION.json")


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def evaluate_forecast_horizon(
    actuals: np.ndarray,
    preds: np.ndarray,
    intervals_80: Optional[List[List[float]]] = None,
    intervals_95: Optional[List[List[float]]] = None
) -> Dict[str, float]:
    """Calculates comprehensive time-series metrics for a forecast horizon."""
    actuals = np.array(actuals)
    preds = np.maximum(np.array(preds), 0.0)

    mae = float(np.mean(np.abs(actuals - preds)))
    rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))
    tot_act = float(np.sum(actuals))
    tot_pred = float(np.sum(preds))

    wape = float(np.sum(np.abs(actuals - preds)) / max(tot_act, 1.0))
    bias = float((tot_pred - tot_act) / max(tot_act, 1.0))
    mape = float(np.mean(np.abs((actuals - preds) / np.maximum(actuals, 1.0))))
    agg_error = float(abs(tot_pred - tot_act) / max(tot_act, 1.0))

    res = {
        "mae": mae,
        "rmse": rmse,
        "wape": wape,
        "mape": mape,
        "bias": bias,
        "aggregated_error": agg_error,
        "actual_sum": tot_act,
        "predicted_sum": tot_pred
    }

    if intervals_80:
        cov_80 = np.mean([low <= act <= high for act, (low, high) in zip(actuals, intervals_80)])
        res["coverage_80"] = float(cov_80)
    if intervals_95:
        cov_95 = np.mean([low <= act <= high for act, (low, high) in zip(actuals, intervals_95)])
        res["coverage_95"] = float(cov_95)

    return res


def run_evaluation_step_7_3b_2() -> Dict[str, Any]:
    print("=" * 80)
    print("STS FINTECH - MILESTONE 2: STEP 7.3B-2 EXPENSE FORECASTER EVALUATION")
    print("=" * 80)

    # 1. Train or load model
    model, train_meta = train_expense_forecaster_v2(random_state=42)
    forecaster = ExpenseForecasterV2()
    history = forecaster._get_history().copy()
    history["date"] = pd.to_datetime(history["date"])

    # 2. Rolling-Origin Evaluation on Historical Holdout (Sep–Oct 2026)
    origins = ["2026-09-01", "2026-09-15", "2026-10-01"]
    horizons = [7, 14, 30]

    rolling_origin_results = {}
    print("\n--- Rolling-Origin Multi-Horizon Historical Evaluation ---")

    for origin in origins:
        origin_dt = pd.to_datetime(origin)
        rolling_origin_results[origin] = {}

        for h in horizons:
            # Generate forecast as of origin date
            fc = forecaster.forecast(horizon_days=h, as_of_date=origin_dt)
            preds = [p["predicted_expenditure_inr"] for p in fc["projections"]]
            int_80 = [p["prediction_interval_80"] for p in fc["projections"]]
            int_95 = [p["prediction_interval_95"] for p in fc["projections"]]

            # Extract actuals for these dates
            dates = [pd.to_datetime(p["date"]) for p in fc["projections"]]
            actual_sub = history[history["date"].isin(dates)].sort_values("date")
            actuals = actual_sub["daily_expenditure"].values

            metrics = evaluate_forecast_horizon(actuals, preds, int_80, int_95)
            rolling_origin_results[origin][f"{h}_day"] = metrics

            print(
                f"  Origin {origin} ({h:2d}-Day) | Daily WAPE: {metrics['wape']*100:5.2f}% | "
                f"Agg Error: {metrics['aggregated_error']*100:5.2f}% | "
                f"Actual: INR {metrics['actual_sum']:11,.2f} | Pred: INR {metrics['predicted_sum']:11,.2f}"
            )

    # Compute average multi-origin metrics
    avg_metrics_by_horizon = {}
    for h in horizons:
        wapes = [rolling_origin_results[o][f"{h}_day"]["wape"] for o in origins]
        agg_errs = [rolling_origin_results[o][f"{h}_day"]["aggregated_error"] for o in origins]
        maes = [rolling_origin_results[o][f"{h}_day"]["mae"] for o in origins]
        biases = [rolling_origin_results[o][f"{h}_day"]["bias"] for o in origins]
        avg_metrics_by_horizon[f"{h}_day"] = {
            "mean_daily_wape": float(np.mean(wapes)),
            "mean_aggregated_error": float(np.mean(agg_errs)),
            "mean_daily_mae": float(np.mean(maes)),
            "mean_bias": float(np.mean(biases))
        }

    print("\n--- Average Across All Rolling Origins ---")
    for h in horizons:
        m = avg_metrics_by_horizon[f"{h}_day"]
        label = "Weekly" if h == 7 else ("Bi-Weekly" if h == 14 else "Monthly")
        print(
            f"  {h:2d}-Day Horizon ({label}) | Mean Daily WAPE: {m['mean_daily_wape']*100:5.2f}% | "
            f"Mean {label} Aggregation Error: {m['mean_aggregated_error']*100:5.2f}% | Mean Bias: {m['mean_bias']*100:5.2f}%"
        )

    # 3. Synthetic Future Stress Test Evaluation (Nov 14 onwards)
    print("\n--- Synthetic Future Stress Test Evaluation (Nov 14 to Dec 14, 2026) ---")
    stress_origin = pd.to_datetime("2026-11-14")
    stress_fc = forecaster.forecast(horizon_days=30, as_of_date=stress_origin)
    stress_preds = [p["predicted_expenditure_inr"] for p in stress_fc["projections"]]
    stress_dates = [pd.to_datetime(p["date"]) for p in stress_fc["projections"]]
    stress_actuals = history[history["date"].isin(stress_dates)].sort_values("date")["daily_expenditure"].values

    stress_metrics = evaluate_forecast_horizon(stress_actuals, stress_preds)
    print(
        f"  Future Stress (30-Day) | Daily WAPE: {stress_metrics['wape']*100:5.2f}% | "
        f"Monthly Agg Error: {stress_metrics['aggregated_error']*100:5.2f}% | "
        f"Actual: INR {stress_metrics['actual_sum']:11,.2f} | Pred: INR {stress_metrics['predicted_sum']:11,.2f}"
    )

    # 4. Generate Exemplar Forecast Schedules for Reporting
    sample_7day = forecaster.forecast(horizon_days=7, as_of_date="2026-10-01")
    sample_30day = forecaster.forecast(horizon_days=30, as_of_date="2026-10-01")

    # 5. Checksum and Metadata Persistence
    model_path = os.path.join(V2_MODELS_DIR, "expense_forecaster_v2.joblib")
    artifact_hash = compute_file_sha256(model_path)
    artifact_size = os.path.getsize(model_path)

    with open(GENERATOR_VERSION_FILE, "r") as f:
        gen_meta = json.load(f)

    full_metadata = {
        "step": "7.3B-2",
        "description": "V2 Continuous Enterprise Expense Forecasting Pipeline",
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "python_version": sys.version,
            "scikit_learn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "joblib_version": joblib.__version__
        },
        "dataset_provenance": {
            "version": gen_meta.get("version", "2.0.0"),
            "generator_script": gen_meta.get("generator_script"),
            "transactions_sha256": gen_meta.get("checksums", {}).get("transactions.csv"),
            "total_transactions": 59834,
            "total_calendar_days": 1096,
            "total_expenditure_inr": float(history["daily_expenditure"].sum())
        },
        "model_artifact": {
            "model_type": train_meta.get("selected_model", "Ridge_Regression"),
            "artifact_path": model_path,
            "sha256": artifact_hash,
            "size_bytes": artifact_size
        },
        "feature_specification": {
            "feature_columns": train_meta.get("feature_columns"),
            "target_variable": "daily_expenditure",
            "leakage_guard": "All lag and rolling features are shifted by >= 1 step. Future actuals are strictly zeroed during recursive rollout."
        },
        "validation_selection": {
            "baselines": train_meta.get("baselines_validation"),
            "candidate_models": train_meta.get("candidate_validation")
        },
        "rolling_origin_evaluation": {
            "origins_evaluated": origins,
            "horizons_evaluated": horizons,
            "detailed_origin_results": rolling_origin_results,
            "average_by_horizon": avg_metrics_by_horizon
        },
        "synthetic_future_stress_evaluation": {
            "origin": "2026-11-14",
            "horizon_days": 30,
            "metrics": stress_metrics
        },
        "prediction_intervals": train_meta.get("prediction_intervals"),
        "exemplar_forecasts": {
            "seven_day": sample_7day,
            "thirty_day": sample_30day
        },
        "synthetic_data_disclaimer": (
            "Forecasts are trained on synthetic enterprise financial data generated for "
            "STS FINTECH PRIVATE LIMITED. Results reflect synthetic seasonality and should be "
            "empirically recalibrated on live customer banking statements."
        )
    }

    with open(METADATA_FILE, "w") as f:
        json.dump(full_metadata, f, indent=2)

    print(f"\nSuccessfully compiled Step 7.3B-2 Metadata to: {METADATA_FILE}")
    print("=" * 80)
    return full_metadata


if __name__ == "__main__":
    run_evaluation_step_7_3b_2()
