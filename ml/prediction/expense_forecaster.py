"""
STS FINTECH AI FINANCIAL INTELLIGENCE
Expense Forecaster Inference & Recursive Projection Engine (Milestone 2 - Step 7.2C)

Generates recursive multi-day forecasts for 7-day, 14-day, and 30-day horizons.
Strictly relies on historical actuals up to the cutoff date, and recursively feeds
prior predictions into subsequent lag and rolling windows without future look-ahead.
Guarantees non-negative expenditure predictions and labels projections after 2026-10-31.
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

from ml.training.train_expense_forecaster import load_and_aggregate_daily_expenses

MODEL_PATH = os.path.join(BASE_DIR, "ml", "models", "expense_forecaster.joblib")
CUTOFF_DATE = pd.to_datetime("2026-10-31").date()


class ExpenseForecaster:
    """
    Inference service for multi-day recursive enterprise expenditure forecasting.
    """

    FEATURE_COLS = [
        "day_of_week",
        "day_of_month",
        "month",
        "quarter",
        "is_weekend",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_30",
        "rolling_mean_7",
        "rolling_std_7",
        "rolling_mean_30",
        "rolling_std_30"
    ]

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model artifact not found at {model_path}. "
                "Please run train_expense_forecaster.py first."
            )
        self.model = joblib.load(model_path)
        self._history = None

    def _get_history(self) -> pd.DataFrame:
        """Loads and caches the daily historical expenditure series."""
        if self._history is None:
            self._history = load_and_aggregate_daily_expenses()
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
            Dict containing forecast schedule, summary totals, and metadata.
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

        # Build rolling window list of expenditures from known history
        history_series = history_df.set_index("date")["daily_expenditure"]
        expenditure_window = list(history_series.values)

        projections = []
        current_date = cutoff

        for step in range(1, horizon_days + 1):
            next_date = current_date + timedelta(days=step)
            next_ts = pd.to_datetime(next_date)

            # 1. Calendar features
            day_of_week = next_ts.dayofweek
            day_of_month = next_ts.day
            month = next_ts.month
            quarter = next_ts.quarter
            is_weekend = int(day_of_week >= 5)

            # 2. Shifted Lags from window
            lag_1 = float(expenditure_window[-1])
            lag_7 = float(expenditure_window[-7]) if len(expenditure_window) >= 7 else lag_1
            lag_14 = float(expenditure_window[-14]) if len(expenditure_window) >= 14 else lag_7
            lag_30 = float(expenditure_window[-30]) if len(expenditure_window) >= 30 else lag_14

            # 3. Shifted Rolling Stats from window
            w7 = expenditure_window[-7:]
            w30 = expenditure_window[-30:]
            rolling_mean_7 = float(np.mean(w7))
            rolling_std_7 = float(np.std(w7)) if len(w7) > 1 else 0.0
            rolling_mean_30 = float(np.mean(w30))
            rolling_std_30 = float(np.std(w30)) if len(w30) > 1 else 0.0

            # 4. Construct single-row feature vector
            feature_vector = pd.DataFrame([{
                "day_of_week": day_of_week,
                "day_of_month": day_of_month,
                "month": month,
                "quarter": quarter,
                "is_weekend": is_weekend,
                "lag_1": lag_1,
                "lag_7": lag_7,
                "lag_14": lag_14,
                "lag_30": lag_30,
                "rolling_mean_7": rolling_mean_7,
                "rolling_std_7": rolling_std_7,
                "rolling_mean_30": rolling_mean_30,
                "rolling_std_30": rolling_std_30
            }])[self.FEATURE_COLS]

            # 5. Predict and enforce non-negative expenditure
            pred_raw = self.model.predict(feature_vector)[0]
            pred_clamped = float(max(0.0, pred_raw))

            # 6. Append prediction to window for subsequent recursive steps
            expenditure_window.append(pred_clamped)

            is_post_cutoff = next_date > CUTOFF_DATE
            projections.append({
                "horizon_step": step,
                "forecast_date": str(next_date),
                "day_name": next_ts.strftime("%A"),
                "is_weekend": bool(is_weekend),
                "predicted_expenditure": round(pred_clamped, 2),
                "projection_status": (
                    "projection_after_dataset_cutoff" if is_post_cutoff else "in_dataset_evaluation"
                )
            })

        total_proj = sum(p["predicted_expenditure"] for p in projections)
        avg_proj = total_proj / horizon_days

        return {
            "forecast_horizon_days": horizon_days,
            "as_of_date": str(cutoff),
            "total_projected_expenditure": round(total_proj, 2),
            "average_daily_projected": round(avg_proj, 2),
            "currency": "INR",
            "projections": projections,
            "disclaimer": (
                "Forecast generated recursively from synthetic enterprise data. "
                "Predictions after 2026-10-31 represent statistical projections "
                "and do not constitute live production financial actuals."
            )
        }
