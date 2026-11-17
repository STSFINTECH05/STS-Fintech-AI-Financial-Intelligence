"""
STS FINTECH AI FINANCIAL INTELLIGENCE
Model C: V2 Expense Forecaster Inference Service (Milestone 2 - Step 7.4B-2C)

Provides multi-day recursive enterprise expenditure forecasting using
the verified RandomForestRegressor artifact.
Enforces:
- Strict validation of input historical series (ISO dates, order, continuity, non-negative finite amounts).
- Currency validation (strictly 'INR').
- Exact minimum historical observation requirement (>= 30 consecutive daily records).
- Horizon validation (1 to 90 days).
- Derivation of forecast origin from request final observation.
- Heuristic planning ranges (never claiming statistical confidence intervals).
- Safe failure handling ('NOT_EVALUATED') when model cannot evaluate.
- Reuses the verified prediction engine in ml.prediction.expense_forecaster_v2.
"""

import math
import time
import logging
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

from ml.service.model_loader import ModelRegistry, APPROVED_MODEL_SPECS
from ml.prediction.expense_forecaster_v2 import ExpenseForecasterV2

logger = logging.getLogger("sts_ml_service.expense_forecaster")

MODEL_ID = "model_c_forecaster"
EXPECTED_SHA256 = APPROVED_MODEL_SPECS[MODEL_ID]["expected_sha256"]
SUPPORTED_CURRENCY = "INR"
MIN_HISTORY_DAYS = 30
MIN_HORIZON_DAYS = 1
MAX_HORIZON_DAYS = 90


class ForecastInputValidationError(ValueError):
    """Raised when forecast input payload fails validation or lacks historical continuity."""
    pass


class ExpenseForecasterService:
    """
    Inference service for Model C (V2 Expense Forecaster).
    Wraps the verified RandomForestRegressor artifact via ExpenseForecasterV2.
    """

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self._model = self.registry.get_model(MODEL_ID)
        if not self._model:
            raise RuntimeError(f"Required model artifact '{MODEL_ID}' is not loaded in registry.")

        # Instantiate prediction engine and bind verified registry model
        self.engine = ExpenseForecasterV2()
        self.engine.model = self._model
        logger.info("ExpenseForecasterService initialized successfully with verified Model C artifact.")

    def _validate_input_payload(self, payload: Dict[str, Any]) -> Tuple[date, int, str, pd.DataFrame]:
        """
        Validates the incoming forecast request payload.
        Returns validated (as_of_date, horizon_days, currency, history_df).
        """
        # 1. Horizon Validation
        if "horizon_days" not in payload or payload["horizon_days"] is None:
            raise ForecastInputValidationError("Missing mandatory field: 'horizon_days'")

        horizon_raw = payload["horizon_days"]
        if not isinstance(horizon_raw, int) or isinstance(horizon_raw, bool):
            raise ForecastInputValidationError(f"Field 'horizon_days' must be an integer. Received: {type(horizon_raw).__name__}")

        if horizon_raw < MIN_HORIZON_DAYS or horizon_raw > MAX_HORIZON_DAYS:
            raise ForecastInputValidationError(
                f"Field 'horizon_days' must be between {MIN_HORIZON_DAYS} and {MAX_HORIZON_DAYS}. Received: {horizon_raw}"
            )
        horizon_days = int(horizon_raw)

        # 2. Currency Validation
        currency = str(payload.get("currency") or "").strip().upper()
        if not currency:
            raise ForecastInputValidationError("Missing mandatory field: 'currency'")
        if currency != SUPPORTED_CURRENCY:
            raise ForecastInputValidationError(
                f"Unsupported currency: '{currency}'. Model C is calibrated exclusively for '{SUPPORTED_CURRENCY}'."
            )

        # 3. as_of_date Validation
        as_of_str = payload.get("as_of_date")
        if not as_of_str or not str(as_of_str).strip():
            raise ForecastInputValidationError("Missing mandatory field: 'as_of_date'")

        try:
            as_of_dt = datetime.strptime(str(as_of_str).strip(), "%Y-%m-%d").date()
        except ValueError:
            raise ForecastInputValidationError(
                f"Invalid date format for 'as_of_date': '{as_of_str}'. Expected ISO format 'YYYY-MM-DD'."
            )

        # 4. Historical Daily Series Validation
        history_raw = payload.get("historical_daily_series")
        if history_raw is None or not isinstance(history_raw, list):
            raise ForecastInputValidationError("Missing mandatory list field: 'historical_daily_series'")

        if len(history_raw) < MIN_HISTORY_DAYS:
            raise ForecastInputValidationError(
                f"Insufficient historical observations. At least {MIN_HISTORY_DAYS} consecutive daily records are required. "
                f"Received: {len(history_raw)}"
            )

        parsed_records: List[Dict[str, Any]] = []
        seen_dates = set()

        for idx, item in enumerate(history_raw):
            if not isinstance(item, dict):
                raise ForecastInputValidationError(f"Item at index {idx} in 'historical_daily_series' must be an object.")

            d_str = item.get("date")
            if not d_str or not str(d_str).strip():
                raise ForecastInputValidationError(f"Item at index {idx} missing required field 'date'.")

            try:
                d_val = datetime.strptime(str(d_str).strip(), "%Y-%m-%d").date()
            except ValueError:
                raise ForecastInputValidationError(
                    f"Item at index {idx} has invalid date format: '{d_str}'. Expected 'YYYY-MM-DD'."
                )

            if d_val in seen_dates:
                raise ForecastInputValidationError(f"Duplicate date detected in historical daily series: {d_val}")
            seen_dates.add(d_val)

            if d_val > as_of_dt:
                raise ForecastInputValidationError(
                    f"Historical observation date {d_val} is in the future relative to as_of_date {as_of_dt}."
                )

            amt_raw = item.get("daily_expenditure")
            if amt_raw is None:
                raise ForecastInputValidationError(f"Item at index {idx} missing required field 'daily_expenditure'.")

            try:
                amt_val = float(amt_raw)
            except (ValueError, TypeError):
                raise ForecastInputValidationError(f"Item at index {idx} 'daily_expenditure' must be numeric.")

            if not math.isfinite(amt_val):
                raise ForecastInputValidationError(
                    f"Item at index {idx} 'daily_expenditure' must be a finite number (not NaN or Infinity)."
                )

            if amt_val < 0.0:
                raise ForecastInputValidationError(
                    f"Item at index {idx} 'daily_expenditure' cannot be negative. Received: {amt_val}"
                )

            parsed_records.append({"date": d_val, "daily_expenditure": amt_val})

        # 5. Chronological Order & Continuity Validation
        for i in range(1, len(parsed_records)):
            prev_d = parsed_records[i - 1]["date"]
            curr_d = parsed_records[i]["date"]

            if curr_d <= prev_d:
                raise ForecastInputValidationError(
                    f"Historical daily series dates must be in strictly ascending order. "
                    f"Date {curr_d} at index {i} is not after previous date {prev_d}."
                )

            if curr_d != prev_d + timedelta(days=1):
                raise ForecastInputValidationError(
                    f"Historical series has missing dates between {prev_d} and {curr_d}. "
                    "All calendar days must be present continuously without gaps."
                )

        # 6. Final Observation matches as_of_date
        final_obs_date = parsed_records[-1]["date"]
        if final_obs_date != as_of_dt:
            raise ForecastInputValidationError(
                f"Final historical observation date ({final_obs_date}) does not match as_of_date ({as_of_dt})."
            )

        history_df = pd.DataFrame(parsed_records)
        return as_of_dt, horizon_days, currency, history_df

    def forecast_expense(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes multi-day expenditure forecasting.
        Returns structured response matching Step 7.4B-2C contract.
        """
        start_time = time.perf_counter()

        as_of_dt, horizon_days, currency, history_df = self._validate_input_payload(payload)
        as_of_iso = as_of_dt.isoformat()

        # Handle uninitialized model
        if self._model is None or self.engine is None or self.engine.model is None:
            logger.error("ExpenseForecasterService invoked with uninitialized model.")
            return {
                "status": "ERROR",
                "inference_status": "NOT_EVALUATED",
                "forecast_origin": as_of_iso,
                "horizon_days": horizon_days,
                "currency": currency,
                "total_projected_expenditure": None,
                "average_daily_expenditure": None,
                "projections": [],
                "planning_range_type": "HEURISTIC_PLANNING_RANGE",
                "provenance": {
                    "model_name": "RandomForestRegressor",
                    "model_version": "2.0.0",
                    "artifact_sha256": EXPECTED_SHA256,
                    "model_training_data_provenance": "SYNTHETIC_V2_DATASET",
                    "inference_latency_ms": 0.0
                }
            }

        # Execute recursive multi-step forecasting via verified engine
        raw_result = self.engine.forecast(
            horizon_days=horizon_days,
            as_of_date=as_of_dt,
            custom_history=history_df
        )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Format projections with HEURISTIC_PLANNING_RANGE labels
        formatted_projections = []
        for proj in raw_result.get("projections", []):
            # Using 80% empirical margin as standard heuristic planning range
            int_range = proj.get("prediction_interval_80", [0.0, 0.0])
            formatted_projections.append({
                "step": proj["step"],
                "date": proj["date"],
                "predicted_expenditure": round(float(proj["predicted_expenditure_inr"]), 2),
                "heuristic_planning_range": {
                    "lower": round(float(int_range[0]), 2),
                    "upper": round(float(int_range[1]), 2)
                }
            })

        total_proj = round(float(raw_result["total_projected_expenditure_inr"]), 2)
        avg_daily = round(float(raw_result["average_daily_expenditure_inr"]), 2)

        logger.info(
            "Forecast generated successfully: origin=%s, horizon=%d, total=%.2f INR, latency=%.2fms",
            as_of_iso,
            horizon_days,
            total_proj,
            elapsed_ms
        )

        return {
            "status": "SUCCESS",
            "inference_status": "EVALUATED",
            "forecast_origin": as_of_iso,
            "horizon_days": horizon_days,
            "currency": currency,
            "total_projected_expenditure": total_proj,
            "average_daily_expenditure": avg_daily,
            "projections": formatted_projections,
            "planning_range_type": "HEURISTIC_PLANNING_RANGE",
            "provenance": {
                "model_name": "RandomForestRegressor",
                "model_version": "2.0.0",
                "artifact_sha256": EXPECTED_SHA256,
                "model_training_data_provenance": "SYNTHETIC_V2_DATASET",
                "inference_latency_ms": elapsed_ms
            }
        }
