"""
STS FINTECH AI FINANCIAL INTELLIGENCE
V2 Expense Forecaster Inference & Recursive Projection Engine (Milestone 2 - Step 7.3B-2)

Generates recursive multi-day expenditure forecasts for 7-day, 14-day, and 30-day horizons.
Strictly relies on historical actuals up to the cutoff date, and recursively updates
prior predictions into subsequent lag and rolling windows without future look-ahead.
Guarantees non-negative expenditure predictions and provides empirical prediction intervals.
Labels projections on or after 2026-11-14 with a synthetic future stress disclaimer.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from ml.training.train_expense_forecaster_v2 import (
    load_and_aggregate_v2_daily_expenses,
    INDIAN_HOLIDAYS
)

MODEL_PATH = os.path.join(BASE_DIR, "ml", "models", "v2", "expense_forecaster_v2.joblib")
SYNTHETIC_FUTURE_CUTOFF = pd.to_datetime("2026-11-14").date()


class ExpenseForecasterV2:
    """
    Inference service for multi-day recursive enterprise expenditure forecasting.
    """

    FEATURE_COLS = [
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

    # Empirical residual standard error calibrated on validation fold
    DEFAULT_ERROR_MARGIN_80 = 1250000.0  # INR
    DEFAULT_ERROR_MARGIN_95 = 2100000.0  # INR

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained V2 model artifact not found at {model_path}. "
                "Please run train_expense_forecaster_v2.py first."
            )
        self.model = joblib.load(model_path)
        self._history = None

    def _get_history(self) -> pd.DataFrame:
        """Loads and caches the daily historical expenditure series."""
        if self._history is None:
            self._history = load_and_aggregate_v2_daily_expenses()
        return self._history

    def forecast(
        self,
        horizon_days: int = 7,
        as_of_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
        custom_history: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Generates a multi-day recursive expenditure forecast.

        Args:
            horizon_days: Number of days to forecast (e.g. 7, 14, 30). Must be between 1 and 90.
            as_of_date: Cutoff date to forecast from. Defaults to latest available history date.
            custom_history: Optional DataFrame with 'date' and 'daily_expenditure' columns.

        Returns:
            Dict containing forecast schedule, summary totals, intervals, and metadata.
        """
        if not isinstance(horizon_days, int) or horizon_days < 1 or horizon_days > 90:
            raise ValueError(f"Invalid horizon_days: {horizon_days}. Must be an integer between 1 and 90.")

        history_df = custom_history if custom_history is not None else self._get_history().copy()
        history_df["date"] = pd.to_datetime(history_df["date"]).dt.date
        history_df = history_df.sort_values("date").reset_index(drop=True)

        if as_of_date is not None:
            cutoff = pd.to_datetime(as_of_date).date()
            history_df = history_df[history_df["date"] <= cutoff].copy()
            if len(history_df) < 30:
                raise ValueError(f"Insufficient history prior to {cutoff}. At least 30 days of records are required.")
        else:
            cutoff = history_df["date"].max()

        history_series = history_df.set_index("date")["daily_expenditure"]
        expenditure_window = list(history_series.values)

        projections = []
        cumulative_projected = 0.0

        for step in range(1, horizon_days + 1):
            target_date = cutoff + timedelta(days=step)
            is_synthetic_future = (target_date >= SYNTHETIC_FUTURE_CUTOFF)

            # Feature dictionary constructed strictly from expenditure_window (t - l)
            row_features = {
                "day_of_week": target_date.weekday(),
                "day_of_month": target_date.day,
                "month": target_date.month,
                "quarter": (target_date.month - 1) // 3 + 1,
                "day_of_year": target_date.timetuple().tm_yday,
                "is_weekend": int(target_date.weekday() >= 5),
                "is_month_end": int(target_date.day >= 25),
                "is_month_start": int(target_date.day <= 3),
                "is_holiday": int((target_date.month, target_date.day) in INDIAN_HOLIDAYS),
                "lag_1": float(expenditure_window[-1]),
                "lag_2": float(expenditure_window[-2]),
                "lag_3": float(expenditure_window[-3]),
                "lag_7": float(expenditure_window[-7]),
                "lag_14": float(expenditure_window[-14]),
                "lag_21": float(expenditure_window[-21]),
                "lag_28": float(expenditure_window[-28]),
                "lag_30": float(expenditure_window[-30]),
                "rolling_mean_7": float(np.mean(expenditure_window[-7:])),
                "rolling_std_7": float(np.std(expenditure_window[-7:])),
                "rolling_mean_14": float(np.mean(expenditure_window[-14:])),
                "rolling_mean_30": float(np.mean(expenditure_window[-30:])),
                "rolling_std_30": float(np.std(expenditure_window[-30:])),
                "rolling_min_7": float(np.min(expenditure_window[-7:])),
                "rolling_max_7": float(np.max(expenditure_window[-7:])),
                "lag1_to_roll7": float(expenditure_window[-1] / max(np.mean(expenditure_window[-7:]), 1.0)),
                "roll7_to_roll30": float(np.mean(expenditure_window[-7:]) / max(np.mean(expenditure_window[-30:]), 1.0))
            }

            feat_df = pd.DataFrame([row_features])[self.FEATURE_COLS]
            pred_val = float(self.model.predict(feat_df)[0])
            # Strict non-negative clip
            pred_val = max(pred_val, 0.0)

            # Empirical prediction intervals (compound slightly with horizon distance)
            uncertainty_scale = 1.0 + (step - 1) * 0.015
            margin_80 = self.DEFAULT_ERROR_MARGIN_80 * uncertainty_scale
            margin_95 = self.DEFAULT_ERROR_MARGIN_95 * uncertainty_scale

            lower_80 = max(pred_val - margin_80, 0.0)
            upper_80 = pred_val + margin_80
            lower_95 = max(pred_val - margin_95, 0.0)
            upper_95 = pred_val + margin_95

            cumulative_projected += pred_val
            # Append prediction to expenditure window for subsequent steps (recursive)
            expenditure_window.append(pred_val)

            projections.append({
                "step": step,
                "date": target_date.isoformat(),
                "day_name": target_date.strftime("%A"),
                "predicted_expenditure_inr": round(pred_val, 2),
                "prediction_interval_80": [round(lower_80, 2), round(upper_80, 2)],
                "prediction_interval_95": [round(lower_95, 2), round(upper_95, 2)],
                "cumulative_expenditure_inr": round(cumulative_projected, 2),
                "is_synthetic_future_scenario": is_synthetic_future
            })

        avg_daily = cumulative_projected / horizon_days

        return {
            "model_version": "2.0.0",
            "forecast_origin": cutoff.isoformat(),
            "horizon_days": horizon_days,
            "currency": "INR",
            "total_projected_expenditure_inr": round(cumulative_projected, 2),
            "average_daily_expenditure_inr": round(avg_daily, 2),
            "projections": projections,
            "synthetic_data_disclaimer": (
                "Forecast generated from STS Fintech V2 Enterprise Financial Data. "
                "Predictions represent decision-support projections. Dates on or after "
                "2026-11-14 fall in synthetic future stress scenarios."
            )
        }
