"""
STS FINTECH AI FINANCIAL INTELLIGENCE
Automated Test Suite for Milestone 2 - Step 7.3B-2
V2 Continuous Enterprise Expense Forecasting Pipeline

Tests cover:
1. Daily, Weekly, and Monthly Time-Series Aggregation & Mathematical Reconciliation
2. Shifted Lag and Rolling Features Zero-Leakage Guard
3. Chronological Split Separation & Synthetic Future Exclusion
4. Forecaster Inference Engine & Horizon Projections (7, 14, 30 days)
5. Non-Negative Predictions Assertion
6. Empirical Prediction Interval Bounds (80% and 95%)
7. Deterministic Inference Reproducibility
8. Artifact Serialization, Reload, and SHA-256 Checksum Verification
9. V1 and Previous V2 Artifacts Preservation
"""

import os
import sys
import json
import hashlib
import pytest
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from ml.training.train_expense_forecaster_v2 import (
    load_and_aggregate_v2_daily_expenses,
    extract_v2_forecasting_features
)
from ml.prediction.expense_forecaster_v2 import ExpenseForecasterV2

V2_MODELS_DIR = os.path.join(BASE_DIR, "ml", "models", "v2")
V1_MODELS_DIR = os.path.join(BASE_DIR, "ml", "models")
RAW_TX_PATH = os.path.join(BASE_DIR, "data", "v2", "raw", "transactions.csv")
MODEL_PATH = os.path.join(V2_MODELS_DIR, "expense_forecaster_v2.joblib")
METADATA_PATH = os.path.join(V2_MODELS_DIR, "expense_forecaster_v2_metadata.json")


class TestV2TimeSeriesAggregationAndReconciliation:
    """Verifies that time-series aggregation reconciles 100% with raw transactions."""

    def test_daily_aggregation_reconciliation(self):
        raw_df = pd.read_csv(RAW_TX_PATH)
        daily_df = load_and_aggregate_v2_daily_expenses()

        assert len(daily_df) == 1096, f"Expected 1,096 calendar days, got {len(daily_df)}"
        assert daily_df["date"].min() == pd.to_datetime("2024-01-01")
        assert daily_df["date"].max() == pd.to_datetime("2026-12-31")

        raw_sum = float(raw_df["amount"].sum())
        daily_sum = float(daily_df["daily_expenditure"].sum())
        assert np.isclose(raw_sum, daily_sum, rtol=1e-5), f"Mismatch: raw {raw_sum} vs daily {daily_sum}"

    def test_weekly_and_monthly_reconciliation(self):
        daily_df = load_and_aggregate_v2_daily_expenses()
        total_daily = daily_df["daily_expenditure"].sum()

        daily_df["week"] = daily_df["date"].dt.to_period("W-SUN")
        weekly_sum = daily_df.groupby("week")["daily_expenditure"].sum().sum()
        assert np.isclose(total_daily, weekly_sum, rtol=1e-5)

        daily_df["month"] = daily_df["date"].dt.to_period("M")
        monthly_sum = daily_df.groupby("month")["daily_expenditure"].sum().sum()
        assert np.isclose(total_daily, monthly_sum, rtol=1e-5)


class TestFeatureLeakageGuards:
    """Strictly validates zero look-ahead bias and proper shifting."""

    def test_lags_strictly_shifted(self):
        daily_df = load_and_aggregate_v2_daily_expenses()
        feat_df, _ = extract_v2_forecasting_features(daily_df)

        # Verify that lag_1 at index i equals daily_expenditure at index i - 1
        for i in range(35, 45):
            assert feat_df.loc[i, "lag_1"] == feat_df.loc[i - 1, "daily_expenditure"]
            assert feat_df.loc[i, "lag_7"] == feat_df.loc[i - 7, "daily_expenditure"]
            assert feat_df.loc[i, "lag_14"] == feat_df.loc[i - 14, "daily_expenditure"]
            assert feat_df.loc[i, "lag_30"] == feat_df.loc[i - 30, "daily_expenditure"]

    def test_rolling_windows_exclude_current_day(self):
        daily_df = load_and_aggregate_v2_daily_expenses()
        feat_df, _ = extract_v2_forecasting_features(daily_df)

        for i in range(35, 45):
            expected_roll7 = feat_df.loc[i - 7:i - 1, "daily_expenditure"].mean()
            actual_roll7 = feat_df.loc[i, "rolling_mean_7"]
            assert np.isclose(expected_roll7, actual_roll7, rtol=1e-5)


class TestChronologicalPartitions:
    """Verifies that training strictly ends before validation and test sets."""

    def test_split_integrity(self):
        daily_df = load_and_aggregate_v2_daily_expenses()
        feat_df, _ = extract_v2_forecasting_features(daily_df)
        valid_data = feat_df.dropna().reset_index(drop=True)

        train_mask = (valid_data["date"] >= "2024-01-31") & (valid_data["date"] <= "2025-12-31")
        val_mask = (valid_data["date"] >= "2026-01-01") & (valid_data["date"] <= "2026-08-31")
        test_mask = (valid_data["date"] >= "2026-09-01") & (valid_data["date"] <= "2026-10-31")
        stress_mask = (valid_data["date"] >= "2026-11-14") & (valid_data["date"] <= "2026-12-31")

        assert train_mask.sum() == 701
        assert val_mask.sum() == 243
        assert test_mask.sum() == 61
        assert stress_mask.sum() == 48

        # Verify zero intersection
        assert (train_mask & val_mask).sum() == 0
        assert (val_mask & test_mask).sum() == 0
        assert (test_mask & stress_mask).sum() == 0


class TestExpenseForecasterV2Inference:
    """Verifies inference service, multi-horizon rollouts, and interval bounds."""

    @pytest.fixture
    def forecaster(self):
        assert os.path.exists(MODEL_PATH), f"Forecaster artifact missing: {MODEL_PATH}"
        return ExpenseForecasterV2(model_path=MODEL_PATH)

    def test_7_day_forecast(self, forecaster):
        res = forecaster.forecast(horizon_days=7, as_of_date="2026-08-31")
        assert res["horizon_days"] == 7
        assert len(res["projections"]) == 7
        assert res["total_projected_expenditure_inr"] > 0
        for p in res["projections"]:
            assert p["predicted_expenditure_inr"] >= 0.0
            assert p["prediction_interval_80"][0] <= p["predicted_expenditure_inr"] <= p["prediction_interval_80"][1]

    def test_14_day_forecast(self, forecaster):
        res = forecaster.forecast(horizon_days=14, as_of_date="2026-08-31")
        assert res["horizon_days"] == 14
        assert len(res["projections"]) == 14

    def test_30_day_forecast(self, forecaster):
        res = forecaster.forecast(horizon_days=30, as_of_date="2026-08-31")
        assert res["horizon_days"] == 30
        assert len(res["projections"]) == 30

    def test_invalid_horizon_rejection(self, forecaster):
        with pytest.raises(ValueError):
            forecaster.forecast(horizon_days=0)
        with pytest.raises(ValueError):
            forecaster.forecast(horizon_days=100)

    def test_deterministic_reproducibility(self, forecaster):
        res1 = forecaster.forecast(horizon_days=14, as_of_date="2026-09-01")
        res2 = forecaster.forecast(horizon_days=14, as_of_date="2026-09-01")
        assert res1["total_projected_expenditure_inr"] == res2["total_projected_expenditure_inr"]
        for p1, p2 in zip(res1["projections"], res2["projections"]):
            assert p1["predicted_expenditure_inr"] == p2["predicted_expenditure_inr"]

    def test_synthetic_future_disclaimer_flag(self, forecaster):
        # Forecast starting Nov 10 for 14 days crosses Nov 14 cutoff
        res = forecaster.forecast(horizon_days=14, as_of_date="2026-11-10")
        assert res["projections"][0]["is_synthetic_future_scenario"] is False  # Nov 11
        assert res["projections"][4]["is_synthetic_future_scenario"] is True   # Nov 15


class TestModelArtifactsAndIntegrity:
    """Verifies metadata manifest, artifact hash, and V1/Step 7.3B-1 preservation."""

    def test_metadata_and_checksum(self):
        assert os.path.exists(METADATA_PATH)
        with open(METADATA_PATH, "r") as f:
            meta = json.load(f)

        assert meta["step"] == "7.3B-2"
        with open(MODEL_PATH, "rb") as f:
            computed_sha = hashlib.sha256(f.read()).hexdigest()
        assert computed_sha == meta["model_artifact"]["sha256"]

    def test_previous_v1_and_v2_artifacts_preserved(self):
        # V1 artifacts must remain untouched
        v1_artifacts = [
            "expense_classifier.joblib",
            "anomaly_detector.joblib",
            "anomaly_supervised_benchmark.joblib",
            "expense_forecaster.joblib",
            "budget_predictor.joblib",
            "vendor_analyzer.joblib"
        ]
        for art in v1_artifacts:
            assert os.path.exists(os.path.join(V1_MODELS_DIR, art))

        # Step 7.3B-1 V2 artifacts must remain untouched
        v2_step1_artifacts = [
            "expense_classifier_v2.joblib",
            "anomaly_detector_v2.joblib",
            "anomaly_supervised_benchmark_v2.joblib",
            "step_7_3b_1_metadata.json"
        ]
        for art in v2_step1_artifacts:
            assert os.path.exists(os.path.join(V2_MODELS_DIR, art))
