"""
STS FINTECH AI FINANCIAL INTELLIGENCE
Model B: V2 Anomaly Detection Inference Service (Milestone 2 - Step 7.4B-2B)

Provides statistical anomaly scoring for financial transactions using
the verified unsupervised IsolationForestV2 artifact.
Enforces:
- Strict mathematical & business validation on all inputs.
- Mandatory presence of real organizational baselines (rejecting fabrication).
- Deterministic ratio derivations matching the training pipeline.
- Normalized anomaly scoring [0.0, 1.0] with calibrated 3-tier risk classification.
- Mandatory human review declaration & prominent disclaimer on every evaluation.
- Safe failure handling ('NOT_EVALUATED') when model cannot evaluate.
"""

import math
import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from ml.service.model_loader import ModelRegistry, APPROVED_MODEL_SPECS

logger = logging.getLogger("sts_ml_service.anomaly_detector")

MODEL_ID = "model_b_anomaly_detector"
EXPECTED_SHA256 = APPROVED_MODEL_SPECS[MODEL_ID]["expected_sha256"]
DISCLAIMER_TEXT = (
    "Unsupervised statistical screening indicator based on synthetic training data. "
    "This is not a determination of fraud, misconduct, or financial policy violation. "
    "Human review is required."
)


class AnomalyInputValidationError(ValueError):
    """Raised when an anomaly inference input fails validation or lacks organizational context."""
    pass


class AnomalyDetectorV2:
    """
    Inference service for Model B (V2 Anomaly Detection).
    Wraps the verified IsolationForestV2AnomalyDetector artifact.
    """

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self._model = self.registry.get_model(MODEL_ID)
        if not self._model:
            raise RuntimeError(f"Required model artifact '{MODEL_ID}' is not loaded in registry.")
        self.threshold_: float = float(getattr(self._model, "threshold_", 0.51))
        logger.info(
            "AnomalyDetectorV2 initialized successfully. Calibrated threshold: %.4f",
            self.threshold_
        )

    def _validate_and_prepare_features(self, payload: Dict[str, Any]) -> pd.DataFrame:
        """
        Validates input fields and derives the exact 24-feature DataFrame
        expected by the ColumnTransformer (14 numeric + 10 categorical).
        """
        # 1. Amount Validation
        if "amount" not in payload or payload["amount"] is None:
            raise AnomalyInputValidationError("Missing mandatory field: 'amount'")

        try:
            amount = float(payload["amount"])
        except (ValueError, TypeError):
            raise AnomalyInputValidationError("Field 'amount' must be a numeric value.")

        if not math.isfinite(amount):
            raise AnomalyInputValidationError("Field 'amount' must be a finite number (not NaN or Infinity).")

        if amount <= 0.0:
            raise AnomalyInputValidationError(f"Field 'amount' must be strictly positive (> 0.0). Received: {amount}")

        log_amount = float(np.log1p(amount))

        # 2. Transaction Date Validation & Calendar Feature Derivation
        date_str = payload.get("transaction_date")
        if not date_str:
            raise AnomalyInputValidationError("Missing mandatory field: 'transaction_date'")

        try:
            dt = pd.to_datetime(date_str, errors="raise")
        except Exception:
            raise AnomalyInputValidationError(
                f"Field 'transaction_date' has an invalid date format: '{date_str}'. Expected valid ISO-8601 date (YYYY-MM-DD)."
            )

        month = int(dt.month)
        day = int(dt.day)
        quarter = int(dt.quarter)
        day_of_week = str(dt.weekday())
        is_weekend = "1" if dt.weekday() >= 5 else "0"

        # 3. Mandatory Categorical Context (Category & Department)
        category = payload.get("category")
        if not category or not str(category).strip():
            raise AnomalyInputValidationError("Missing or empty mandatory field: 'category'")
        category_str = str(category).strip()

        department_id = payload.get("department_id")
        if not department_id or not str(department_id).strip():
            raise AnomalyInputValidationError("Missing or empty mandatory field: 'department_id'")
        department_id_str = str(department_id).strip()

        # 4. Mandatory Organizational Baselines (Strictly rejecting fabrication)
        if "category_median" not in payload or payload["category_median"] is None:
            raise AnomalyInputValidationError(
                "Missing required organizational context: 'category_median'. "
                "Cannot evaluate anomaly risk without baseline category spending median."
            )
        try:
            category_median = float(payload["category_median"])
            if not math.isfinite(category_median) or category_median <= 0.0:
                raise ValueError()
        except (ValueError, TypeError):
            raise AnomalyInputValidationError(
                "Field 'category_median' must be a finite positive number (> 0.0)."
            )

        if "department_median" not in payload or payload["department_median"] is None:
            raise AnomalyInputValidationError(
                "Missing required organizational context: 'department_median'. "
                "Cannot evaluate anomaly risk without baseline department spending median."
            )
        try:
            department_median = float(payload["department_median"])
            if not math.isfinite(department_median) or department_median <= 0.0:
                raise ValueError()
        except (ValueError, TypeError):
            raise AnomalyInputValidationError(
                "Field 'department_median' must be a finite positive number (> 0.0)."
            )

        if "spending_approval_limit" not in payload or payload["spending_approval_limit"] is None:
            raise AnomalyInputValidationError(
                "Missing required organizational context: 'spending_approval_limit'. "
                "Cannot evaluate split threshold anomaly patterns without employee approval limit."
            )
        try:
            spending_approval_limit = float(payload["spending_approval_limit"])
            if not math.isfinite(spending_approval_limit) or spending_approval_limit <= 0.0:
                raise ValueError()
        except (ValueError, TypeError):
            raise AnomalyInputValidationError(
                "Field 'spending_approval_limit' must be a finite positive number (> 0.0)."
            )

        if "annual_budget_baseline" not in payload or payload["annual_budget_baseline"] is None:
            raise AnomalyInputValidationError(
                "Missing required organizational context: 'annual_budget_baseline'."
            )
        try:
            annual_budget_baseline = float(payload["annual_budget_baseline"])
            if not math.isfinite(annual_budget_baseline) or annual_budget_baseline < 0.0:
                raise ValueError()
        except (ValueError, TypeError):
            raise AnomalyInputValidationError(
                "Field 'annual_budget_baseline' must be a finite non-negative number."
            )

        if "vendor_rating" not in payload or payload["vendor_rating"] is None:
            raise AnomalyInputValidationError(
                "Missing required organizational context: 'vendor_rating'."
            )
        try:
            vendor_rating = float(payload["vendor_rating"])
            if not math.isfinite(vendor_rating) or not (1.0 <= vendor_rating <= 5.0):
                raise ValueError()
        except (ValueError, TypeError):
            raise AnomalyInputValidationError(
                "Field 'vendor_rating' must be a finite number between 1.0 and 5.0."
            )

        # 5. Deterministic Ratio & Signal Derivations matching training pipeline
        amount_to_category_median = float(amount / max(category_median, 1.0))
        amount_to_dept_median = float(amount / max(department_median, 1.0))

        spend_limit_clamped = max(spending_approval_limit, 1000.0)
        amount_to_spending_limit = float(amount / spend_limit_clamped)

        near_approval_threshold = 1 if (0.90 <= amount_to_spending_limit <= 1.00) else 0
        is_low_rated_vendor = 1 if (vendor_rating < 2.5) else 0

        # 6. Optional Operational Fields with Verified Training Defaults
        payment_terms_val = payload.get("payment_terms_days", 30)
        payment_terms_days = int(payment_terms_val if payment_terms_val is not None else 30)
        if payment_terms_days < 0:
            raise AnomalyInputValidationError("Field 'payment_terms_days' must be non-negative.")

        vendor_category = str(payload.get("vendor_category") or "General Supplier").strip()
        division = str(payload.get("division") or "General Operations").strip()
        payment_method = str(payload.get("payment_method") or "OTHER").strip()
        transaction_type = str(payload.get("transaction_type") or "PURCHASE").strip()
        employment_type = str(payload.get("employment_type") or "FULL_TIME").strip()

        pref_raw = payload.get("is_preferred_vendor", False)
        if isinstance(pref_raw, str):
            is_pref_str = "1" if pref_raw.strip().lower() in ("true", "1", "yes") else "0"
        else:
            is_pref_str = "1" if bool(pref_raw) else "0"

        # 7. Construct exact 24-column DataFrame matching Model B ColumnTransformer
        return pd.DataFrame({
            # 14 Numeric features
            "amount": [amount],
            "log_amount": [log_amount],
            "amount_to_category_median": [amount_to_category_median],
            "amount_to_dept_median": [amount_to_dept_median],
            "amount_to_spending_limit": [amount_to_spending_limit],
            "near_approval_threshold": [near_approval_threshold],
            "is_low_rated_vendor": [is_low_rated_vendor],
            "vendor_rating": [vendor_rating],
            "payment_terms_days": [payment_terms_days],
            "annual_budget_baseline": [annual_budget_baseline],
            "spending_approval_limit": [spending_approval_limit],
            "month": [month],
            "day": [day],
            "quarter": [quarter],
            # 10 Categorical features
            "category": [category_str],
            "vendor_category": [vendor_category],
            "division": [division],
            "department_id": [department_id_str],
            "payment_method": [payment_method],
            "transaction_type": [transaction_type],
            "is_preferred_vendor": [is_pref_str],
            "employment_type": [employment_type],
            "day_of_week": [day_of_week],
            "is_weekend": [is_weekend]
        })

    def score_transaction(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes anomaly detection scoring.

        Returns structured response conforming strictly to Step 7.4B-2B contract.
        """
        start_time = time.perf_counter()

        # If model is uninitialized or missing
        if self._model is None:
            logger.error("AnomalyDetectorV2 invoked with uninitialized model binary.")
            return {
                "status": "UNAVAILABLE",
                "inference_status": "NOT_EVALUATED",
                "anomaly_indicator": 0,
                "anomaly_score": None,
                "calibrated_threshold": self.threshold_,
                "advisory_classification": "NOT_EVALUATED",
                "review_recommended": True,
                "mandatory_human_review": True,
                "disclaimer": DISCLAIMER_TEXT,
                "provenance": {
                    "model_version": "2.0.0",
                    "model_name": "IsolationForestV2AnomalyDetector",
                    "artifact_sha256": EXPECTED_SHA256,
                    "data_provenance": "SYNTHETIC_V2_DATASET",
                    "inference_latency_ms": 0.0
                }
            }

        # Validate and prepare features
        X = self._validate_and_prepare_features(payload)

        # Execute model inference
        raw_score = float(self._model.predict_anomaly_score(X)[0])
        indicator = int(self._model.predict(X)[0])
        risk_tier = str(self._model.predict_risk_tier(X)[0])

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Review recommended if indicator is 1 or discrepancy tier is moderate/high
        review_recommended = bool(indicator == 1 or risk_tier != "Low Variance")

        logger.info(
            "Anomaly scored successfully: score=%.4f, indicator=%d, tier='%s', latency=%.2fms",
            raw_score,
            indicator,
            risk_tier,
            elapsed_ms
        )

        return {
            "status": "SUCCESS",
            "inference_status": "EVALUATED",
            "anomaly_indicator": indicator,
            "anomaly_score": round(raw_score, 4),
            "calibrated_threshold": self.threshold_,
            "advisory_classification": risk_tier,
            "review_recommended": review_recommended,
            "mandatory_human_review": True,
            "disclaimer": DISCLAIMER_TEXT,
            "provenance": {
                "model_version": "2.0.0",
                "model_name": "IsolationForestV2AnomalyDetector",
                "artifact_sha256": EXPECTED_SHA256,
                "data_provenance": "SYNTHETIC_V2_DATASET",
                "inference_latency_ms": elapsed_ms
            }
        }
