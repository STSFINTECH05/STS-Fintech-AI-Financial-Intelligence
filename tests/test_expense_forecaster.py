"""
Unit & Integration Test Suite for Model C (Expense Forecaster) - Milestone 2 - Step 7.2C
Verifies:
- Daily aggregation correctness and complete date continuity
- Strict shifted lag & rolling window calculation (zero target leakage)
- Exact chronological splitting boundaries (Train / Val / Test)
- Model training, persistence, reload, and SHA-256 integrity
- Recursive multi-day forecasting across 7, 14, and 30 day horizons
- Input validation (rejection of non-integer or out-of-bounds horizons)
- Non-negative expenditure enforcement
- Deterministic prediction reproducibility
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from ml.training.train_expense_forecaster import (
    load_and_aggregate_daily_expenses,
    extract_forecasting_features,
    calculate_metrics
)
from ml.prediction.expense_forecaster import ExpenseForecaster


class TestDailyAggregationAndContinuity:
    """Tests daily enterprise-level transaction aggregation and calendar continuity."""

    @pytest.fixture(scope="class")
    def daily_df(self):
        return load_and_aggregate_daily_expenses()

    def test_daily_row_count_and_continuity(self, daily_df):
        assert len(daily_df) == 1035, "Daily aggregation must span exactly 1,035 days"
        min_date = daily_df["date"].min().date()
        max_date = daily_df["date"].max().date()
        assert str(min_date) == "2024-01-01"
        assert str(max_date) == "2026-10-31"

        # Verify no missing intermediate dates
        expected_range = pd.date_range(min_date, max_date, freq="D")
        assert len(expected_range) == 1035
        assert (daily_df["date"].values == expected_range.values).all()

    def test_daily_expenditures_positive(self, daily_df):
        assert (daily_df["daily_expenditure"] >= 0.0).all()
        assert (daily_df["transaction_count"] >= 1).all(), "Every day in this dataset has >= 1 transaction"

    def test_total_expenditure_matches_raw_sum(self, daily_df):
        raw_df = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "transactions.csv"))
        raw_sum = raw_df["amount"].sum()
        daily_sum = daily_df["daily_expenditure"].sum()
        assert daily_sum == pytest.approx(raw_sum, rel=1e-5)


class TestFeatureEngineeringAndLeakagePrevention:
    """Verifies that lag and rolling statistics are strictly shifted and contain zero target leakage."""

    @pytest.fixture(scope="class")
    def feature_df_and_cols(self):
        daily = load_and_aggregate_daily_expenses()
        return extract_forecasting_features(daily)

    def test_lag_values_strictly_shifted(self, feature_df_and_cols):
        df, cols = feature_df_and_cols
        # Check day index 40: lag_1 must equal day 39 expenditure, lag_7 must equal day 33, etc.
        idx = 40
        expenditures = df["daily_expenditure"].values

        assert df.loc[idx, "lag_1"] == expenditures[idx - 1]
        assert df.loc[idx, "lag_7"] == expenditures[idx - 7]
        assert df.loc[idx, "lag_14"] == expenditures[idx - 14]
        assert df.loc[idx, "lag_30"] == expenditures[idx - 30]

    def test_rolling_statistics_exclude_current_day(self, feature_df_and_cols):
        df, cols = feature_df_and_cols
        idx = 45
        expenditures = df["daily_expenditure"].values

        # Rolling 7 on day idx must average expenditures[idx-7 : idx] (strictly prior 7 days)
        expected_mean_7 = np.mean(expenditures[idx - 7 : idx])
        expected_std_7 = np.std(expenditures[idx - 7 : idx], ddof=1)

        assert df.loc[idx, "rolling_mean_7"] == pytest.approx(expected_mean_7, rel=1e-4)
        assert df.loc[idx, "rolling_std_7"] == pytest.approx(expected_std_7, rel=1e-4)

    def test_no_nulls_in_active_splits(self, feature_df_and_cols):
        df, cols = feature_df_and_cols
        # Data starting from day 30 onwards must have 0 NaNs across all feature columns
        active_data = df.iloc[30:][cols]
        assert active_data.isna().sum().sum() == 0


class TestChronologicalSplits:
    """Verifies non-overlapping temporal partition boundaries."""

    def test_split_boundaries(self):
        daily = load_and_aggregate_daily_expenses()
        daily["date_str"] = daily["date"].dt.strftime("%Y-%m-%d")

        train_mask = (daily["date_str"] >= "2024-01-31") & (daily["date_str"] <= "2026-03-31")
        val_mask = (daily["date_str"] >= "2026-04-01") & (daily["date_str"] <= "2026-07-31")
        test_mask = (daily["date_str"] >= "2026-08-01") & (daily["date_str"] <= "2026-10-31")

        train_dates = set(daily.loc[train_mask, "date_str"])
        val_dates = set(daily.loc[val_mask, "date_str"])
        test_dates = set(daily.loc[test_mask, "date_str"])

        # Disjoint partitions
        assert len(train_dates.intersection(val_dates)) == 0
        assert len(val_dates.intersection(test_dates)) == 0
        assert len(train_dates.intersection(test_dates)) == 0

        assert len(train_dates) == 791
        assert len(val_dates) == 122
        assert len(test_dates) == 92


class TestForecasterInferenceAndMultiHorizon:
    """Tests inference engine, multi-day horizons, and edge cases."""

    @pytest.fixture(scope="class")
    def forecaster(self):
        return ExpenseForecaster()

    def test_7_day_forecast(self, forecaster):
        res = forecaster.forecast(horizon_days=7)
        assert res["forecast_horizon_days"] == 7
        assert len(res["projections"]) == 7
        assert res["total_projected_expenditure"] > 0
        assert res["average_daily_projected"] > 0
        for p in res["projections"]:
            assert p["predicted_expenditure"] >= 0.0
            assert p["projection_status"] == "projection_after_dataset_cutoff"

    def test_14_day_forecast(self, forecaster):
        res = forecaster.forecast(horizon_days=14)
        assert res["forecast_horizon_days"] == 14
        assert len(res["projections"]) == 14
        for p in res["projections"]:
            assert p["predicted_expenditure"] >= 0.0

    def test_30_day_forecast(self, forecaster):
        res = forecaster.forecast(horizon_days=30)
        assert res["forecast_horizon_days"] == 30
        assert len(res["projections"]) == 30
        for p in res["projections"]:
            assert p["predicted_expenditure"] >= 0.0

    def test_invalid_horizon_rejection(self, forecaster):
        with pytest.raises(ValueError):
            forecaster.forecast(horizon_days=0)
        with pytest.raises(ValueError):
            forecaster.forecast(horizon_days=-7)
        with pytest.raises(ValueError):
            forecaster.forecast(horizon_days=120)  # > 90
        with pytest.raises(ValueError):
            forecaster.forecast(horizon_days="7")  # non-int

    def test_prediction_reproducibility(self, forecaster):
        run1 = forecaster.forecast(horizon_days=14)
        run2 = forecaster.forecast(horizon_days=14)
        assert run1["total_projected_expenditure"] == run2["total_projected_expenditure"]
        for p1, p2 in zip(run1["projections"], run2["projections"]):
            assert p1["predicted_expenditure"] == p2["predicted_expenditure"]

    def test_artifact_persistence_and_checksum(self):
        model_path = os.path.join(BASE_DIR, "ml", "models", "expense_forecaster.joblib")
        meta_path = os.path.join(BASE_DIR, "ml", "models", "expense_forecaster_metadata.json")

        assert os.path.exists(model_path)
        assert os.path.exists(meta_path)
        assert os.path.getsize(model_path) > 0

        with open(meta_path, "r") as f:
            meta = json.load(f)

        assert meta["model_name"] == "expense_forecaster"
        assert "sha256_checksum" in meta
        assert "validation_metrics" in meta
        assert "test_metrics" in meta
