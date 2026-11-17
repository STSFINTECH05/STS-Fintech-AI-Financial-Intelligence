"""
STS FINTECH AI FINANCIAL INTELLIGENCE
Model D: V2 Budget Utilization Prediction Service (Milestone 2 - Step 7.4B-2D)

Provides departmental budget utilization forecasting, final monthly expenditure
estimation, remaining allocation, and deterministic overrun risk screening using
the verified RandomForestRegressor bundle.
Enforces:
- Strict validation of snapshot inputs (department_id, year, month, as_of_day in [7, 14, 21]).
- Strict validation of financial amounts (allocated_amount > 0, month_to_date_actual >= 0, finite).
- Derivation of the exact 16 training features in identical order.
- Deterministic 3-tier advisory risk classification without overrun probability conversion.
- Safe error handling ('NOT_EVALUATED') when model cannot evaluate.
- Cryptographic provenance and latency measurement.
"""

import math
import time
import calendar
import logging
from typing import Dict, Any, List, Optional, Union
import pandas as pd

from ml.service.model_loader import ModelRegistry, APPROVED_MODEL_SPECS

logger = logging.getLogger("sts_ml_service.budget_predictor")

MODEL_ID = "model_d_budget_predictor"
EXPECTED_SHA256 = APPROVED_MODEL_SPECS[MODEL_ID]["expected_sha256"]
SUPPORTED_SNAPSHOT_DAYS = {7, 14, 21}
ADVISORY_NOTICE = "Informational spending pace indicator only. Human financial review required."

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


class BudgetInputValidationError(ValueError):
    """Raised when budget snapshot inputs fail validation or specify unsupported horizons."""
    pass


class BudgetPredictorV2Service:
    """
    Inference service for Model D (V2 Budget Utilization Predictor).
    Wraps the verified RandomForestRegressor bundle artifact.
    """

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        bundle = self.registry.get_model(MODEL_ID)
        if not bundle:
            raise RuntimeError(f"Required model artifact '{MODEL_ID}' is not loaded in registry.")

        if isinstance(bundle, dict):
            self.model = bundle["model"]
            self.feature_names = bundle.get("feature_names", FEATURE_COLS)
            self.department_profiles = bundle.get("department_profiles", {})
            self.model_name = bundle.get("model_name", "RandomForestRegressor")
        else:
            self.model = bundle
            self.feature_names = FEATURE_COLS
            self.department_profiles = {}
            self.model_name = "RandomForestRegressor"

        logger.info("BudgetPredictorV2Service initialized successfully with verified Model D artifact.")

    def _validate_and_prepare_features(self, payload: Dict[str, Any]) -> pd.DataFrame:
        """
        Validates input fields and derives the exact 16 features for Model D.
        """
        # 1. Department ID Validation
        dept_raw = payload.get("department_id")
        if dept_raw is None or str(dept_raw).strip() == "":
            raise BudgetInputValidationError("Missing mandatory field: 'department_id'")

        try:
            dept_id = int(dept_raw)
            if dept_id <= 0:
                raise ValueError()
        except (ValueError, TypeError):
            raise BudgetInputValidationError(
                f"Field 'department_id' must be a positive integer or numeric string (e.g., 1 to 50). Received: {dept_raw}"
            )

        # 2. Year and Month Validation
        year_raw = payload.get("budget_year")
        if year_raw is None:
            raise BudgetInputValidationError("Missing mandatory field: 'budget_year'")
        try:
            budget_year = int(year_raw)
            if budget_year < 2000 or budget_year > 2100:
                raise ValueError()
        except (ValueError, TypeError):
            raise BudgetInputValidationError(f"Field 'budget_year' must be a valid four-digit year (2000-2100). Received: {year_raw}")

        month_raw = payload.get("budget_month")
        if month_raw is None:
            raise BudgetInputValidationError("Missing mandatory field: 'budget_month'")
        try:
            budget_month = int(month_raw)
            if budget_month < 1 or budget_month > 12:
                raise ValueError()
        except (ValueError, TypeError):
            raise BudgetInputValidationError(f"Field 'budget_month' must be an integer between 1 and 12. Received: {month_raw}")

        days_in_month = calendar.monthrange(budget_year, budget_month)[1]

        # 3. Snapshot Horizon Validation
        day_raw = payload.get("as_of_day")
        if day_raw is None:
            raise BudgetInputValidationError("Missing mandatory field: 'as_of_day'")
        try:
            as_of_day = int(day_raw)
        except (ValueError, TypeError):
            raise BudgetInputValidationError(f"Field 'as_of_day' must be an integer. Received: {day_raw}")

        if as_of_day < 1 or as_of_day > days_in_month:
            raise BudgetInputValidationError(
                f"Field 'as_of_day' must be between 1 and {days_in_month} for month {budget_month}/{budget_year}. Received: {as_of_day}"
            )

        if as_of_day not in SUPPORTED_SNAPSHOT_DAYS:
            raise BudgetInputValidationError(
                f"Unsupported snapshot horizon: as_of_day must be one of {sorted(list(SUPPORTED_SNAPSHOT_DAYS))}. Received: {as_of_day}"
            )

        # 4. Allocated Amount Validation
        alloc_raw = payload.get("allocated_amount")
        if alloc_raw is None:
            raise BudgetInputValidationError("Missing mandatory field: 'allocated_amount'")
        try:
            allocated_amount = float(alloc_raw)
        except (ValueError, TypeError):
            raise BudgetInputValidationError("Field 'allocated_amount' must be numeric.")

        if not math.isfinite(allocated_amount):
            raise BudgetInputValidationError("Field 'allocated_amount' must be a finite number (not NaN or Infinity).")

        if allocated_amount <= 0.0:
            raise BudgetInputValidationError(
                f"Field 'allocated_amount' must be strictly positive (> 0.0). Received: {allocated_amount}"
            )

        # 5. Month-to-Date Actual Validation
        actual_raw = payload.get("month_to_date_actual")
        if actual_raw is None:
            raise BudgetInputValidationError("Missing mandatory field: 'month_to_date_actual'")
        try:
            month_to_date_actual = float(actual_raw)
        except (ValueError, TypeError):
            raise BudgetInputValidationError("Field 'month_to_date_actual' must be numeric.")

        if not math.isfinite(month_to_date_actual):
            raise BudgetInputValidationError("Field 'month_to_date_actual' must be a finite number (not NaN or Infinity).")

        if month_to_date_actual < 0.0:
            raise BudgetInputValidationError(
                f"Field 'month_to_date_actual' must be non-negative (>= 0.0). Received: {month_to_date_actual}"
            )

        # 6. Optional Transaction Count and Baseline
        tx_count_raw = payload.get("transaction_count_mtd")
        if tx_count_raw is not None:
            try:
                tx_count = int(tx_count_raw)
                if tx_count < 0:
                    raise ValueError()
            except (ValueError, TypeError):
                raise BudgetInputValidationError("Field 'transaction_count_mtd' must be a non-negative integer.")
        else:
            tx_count = max(1, int(round(month_to_date_actual / 15000.0)))

        dept_profile = self.department_profiles.get(dept_id, {
            "hist_mean_util": 85.0,
            "hist_median_util": 85.0,
            "hist_overrun_freq": 0.15,
            "annual_budget_baseline": allocated_amount * 12.0
        })

        baseline_raw = payload.get("annual_budget_baseline")
        if baseline_raw is not None:
            try:
                annual_budget_baseline = float(baseline_raw)
                if not math.isfinite(annual_budget_baseline) or annual_budget_baseline <= 0:
                    raise ValueError()
            except (ValueError, TypeError):
                raise BudgetInputValidationError("Field 'annual_budget_baseline' must be a positive finite number.")
        else:
            annual_budget_baseline = dept_profile.get("annual_budget_baseline", allocated_amount * 12.0)

        # 7. Exact Feature Derivations matching training
        elapsed_pct = (as_of_day / days_in_month) * 100.0
        remaining_days = days_in_month - as_of_day
        linear_runrate = month_to_date_actual / max(1e-4, (elapsed_pct / 100.0))
        mtd_util_pct = (month_to_date_actual / allocated_amount) * 100.0
        linear_util_pct = (linear_runrate / allocated_amount) * 100.0
        tx_rate_per_day = tx_count / max(1, as_of_day)
        avg_tx_size = month_to_date_actual / max(1, tx_count)
        alloc_vs_baseline = allocated_amount / max(1.0, (annual_budget_baseline / 12.0))
        remaining_budget_mtd = allocated_amount - month_to_date_actual

        feature_row = {
            "department_id": dept_id,
            "budget_month": int(budget_month),
            "as_of_day": int(as_of_day),
            "elapsed_month_pct": float(elapsed_pct),
            "allocated_amount": float(allocated_amount),
            "month_to_date_actual": float(month_to_date_actual),
            "transaction_count_mtd": int(tx_count),
            "days_in_month": int(days_in_month),
            "remaining_days": int(remaining_days),
            "linear_runrate": float(linear_runrate),
            "mtd_utilization_pct": float(mtd_util_pct),
            "linear_runrate_utilization_pct": float(linear_util_pct),
            "tx_rate_per_day": float(tx_rate_per_day),
            "avg_tx_size_mtd": float(avg_tx_size),
            "alloc_vs_baseline": float(alloc_vs_baseline),
            "remaining_budget_mtd": float(remaining_budget_mtd)
        }

        return pd.DataFrame([feature_row])[self.feature_names]

    def predict_budget(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes budget utilization prediction.
        Returns structured response conforming to Step 7.4B-2D contract.
        """
        start_time = time.perf_counter()

        # Handle uninitialized model
        if self.model is None:
            logger.error("BudgetPredictorV2Service invoked with uninitialized model binary.")
            return {
                "status": "ERROR",
                "inference_status": "NOT_EVALUATED",
                "department_id": str(payload.get("department_id", "")),
                "budget_year": int(payload.get("budget_year", 2026)) if payload.get("budget_year") else 2026,
                "budget_month": int(payload.get("budget_month", 1)) if payload.get("budget_month") else 1,
                "as_of_day": int(payload.get("as_of_day", 14)) if payload.get("as_of_day") else 14,
                "allocated_amount": float(payload.get("allocated_amount", 0.0)) if payload.get("allocated_amount") else 0.0,
                "month_to_date_actual": float(payload.get("month_to_date_actual", 0.0)) if payload.get("month_to_date_actual") else 0.0,
                "predicted_final_actual": None,
                "predicted_utilization_pct": None,
                "estimated_remaining_budget": None,
                "advisory_risk_tier": "NOT_EVALUATED",
                "advisory_notice": ADVISORY_NOTICE,
                "provenance": {
                    "model_name": "RandomForestRegressor",
                    "model_version": "2.0.0",
                    "artifact_sha256": EXPECTED_SHA256,
                    "data_provenance": "SYNTHETIC_V2_DATASET",
                    "model_training_data_provenance": "SYNTHETIC_V2_DATASET",
                    "inference_latency_ms": 0.0
                }
            }

        # Prepare and validate features
        X_df = self._validate_and_prepare_features(payload)

        dept_id = int(payload["department_id"])
        budget_year = int(payload["budget_year"])
        budget_month = int(payload["budget_month"])
        as_of_day = int(payload["as_of_day"])
        allocated_amount = float(payload["allocated_amount"])
        month_to_date_actual = float(payload["month_to_date_actual"])

        # Execute model prediction
        pred_util = float(self.model.predict(X_df)[0])
        mtd_util_pct = (month_to_date_actual / allocated_amount) * 100.0

        # Monotonic constraint: final utilization cannot be lower than current MTD utilization
        pred_util = max(mtd_util_pct, max(0.0, pred_util))
        pred_actual = float(round(allocated_amount * (pred_util / 100.0), 2))
        est_remaining = float(round(allocated_amount - pred_actual, 2))
        pred_util = float(round(pred_util, 2))

        # Deterministic Advisory Tiers (strictly NO overrun probability)
        if pred_util > 100.0:
            advisory_tier = "Potential Budget Overrun – Requires Verification"
        elif pred_util >= 90.0:
            advisory_tier = "Approaching Budget Limit"
        else:
            advisory_tier = "Within Budget"

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.info(
            "Budget predicted: dept=%d, as_of_day=%d, util=%.2f%%, actual=%.2f, tier='%s', latency=%.2fms",
            dept_id,
            as_of_day,
            pred_util,
            pred_actual,
            advisory_tier,
            elapsed_ms
        )

        return {
            "status": "SUCCESS",
            "inference_status": "EVALUATED",
            "department_id": str(dept_id),
            "budget_year": budget_year,
            "budget_month": budget_month,
            "as_of_day": as_of_day,
            "allocated_amount": allocated_amount,
            "month_to_date_actual": month_to_date_actual,
            "predicted_final_actual": pred_actual,
            "predicted_utilization_pct": pred_util,
            "estimated_remaining_budget": est_remaining,
            "advisory_risk_tier": advisory_tier,
            "advisory_notice": ADVISORY_NOTICE,
            "provenance": {
                "model_name": "RandomForestRegressor",
                "model_version": "2.0.0",
                "artifact_sha256": EXPECTED_SHA256,
                "data_provenance": "SYNTHETIC_V2_DATASET",
                "model_training_data_provenance": "SYNTHETIC_V2_DATASET",
                "inference_latency_ms": elapsed_ms
            }
        }
