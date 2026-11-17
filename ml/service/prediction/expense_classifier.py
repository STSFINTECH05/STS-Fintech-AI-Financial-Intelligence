"""
STS FINTECH AI FINANCIAL INTELLIGENCE
Model A: V2 Expense Classification Inference Service (Milestone 2 - Step 7.4B-2A)

Provides low-latency, deterministic expense classification inference using
the cryptographically verified Model A artifact (LogisticRegression pipeline).
Enforces:
- Strict mathematical & business validation on all inputs.
- Deterministic feature derivation matching the original training pipeline.
- Raw uncalibrated decision scores (explicitly barring claims of confidence/accuracy).
- Mandatory manual confirmation flag on every response.
- Sanitized logging (zero leakage of transaction descriptions or PII).
"""

import math
import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from ml.service.model_loader import ModelRegistry, APPROVED_MODEL_SPECS

logger = logging.getLogger("sts_ml_service.expense_classifier")

MODEL_ID = "model_a_classifier"
EXPECTED_SHA256 = APPROVED_MODEL_SPECS[MODEL_ID]["expected_sha256"]


class InputValidationError(ValueError):
    """Raised when an inference input fails business or numerical validation."""
    pass


class ExpenseClassifierV2:
    """
    Inference service for Model A (V2 Expense Classification).
    Wraps the verified scikit-learn Pipeline and executes predictions.
    """

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self._model = self.registry.get_model(MODEL_ID)
        if not self._model:
            raise RuntimeError(f"Required model artifact '{MODEL_ID}' is not loaded in registry.")
        self.classes_: List[str] = list(self._model.classes_)
        logger.info(
            "ExpenseClassifierV2 initialized successfully with %d categories.",
            len(self.classes_)
        )

    def _validate_and_prepare_features(self, payload: Dict[str, Any]) -> pd.DataFrame:
        """
        Validates input fields and derives the exact 18-feature DataFrame
        expected by the trained ColumnTransformer pipeline.
        """
        # 1. Amount Validation
        if "amount" not in payload or payload["amount"] is None:
            raise InputValidationError("Missing mandatory field: 'amount'")

        try:
            amount = float(payload["amount"])
        except (ValueError, TypeError):
            raise InputValidationError("Field 'amount' must be a numeric value.")

        if not math.isfinite(amount):
            raise InputValidationError("Field 'amount' must be a finite number (not NaN or Infinity).")

        if amount <= 0.0:
            raise InputValidationError(f"Field 'amount' must be strictly positive (> 0.0). Received: {amount}")

        # Derive log_amount exactly as in training
        log_amount = float(np.log1p(amount))

        # 2. Description Validation (Max 255 chars, handle empty/missing)
        raw_desc = payload.get("description")
        if raw_desc is None:
            description = ""
        else:
            description = str(raw_desc).strip()
            if len(description) > 255:
                raise InputValidationError(
                    f"Field 'description' exceeds maximum allowed length of 255 characters (length: {len(description)})."
                )

        # 3. Transaction Date Validation & Calendar Feature Derivation
        date_str = payload.get("transaction_date")
        if not date_str:
            raise InputValidationError("Missing mandatory field: 'transaction_date'")

        try:
            # Enforce strict parsing
            dt = pd.to_datetime(date_str, errors="raise")
        except Exception:
            raise InputValidationError(
                f"Field 'transaction_date' has an invalid date format: '{date_str}'. Expected valid ISO-8601 date (YYYY-MM-DD)."
            )

        # Calendar derivations matching data_cleaner_v2.py
        month = int(dt.month)
        day = int(dt.day)
        quarter = int(dt.quarter)
        day_of_week = int(dt.weekday())  # 0=Monday .. 6=Sunday
        is_weekend = 1 if day_of_week >= 5 else 0

        # 4. Mandatory Department ID Validation (No synthetic zero-fallback)
        department_id = payload.get("department_id")
        if department_id is None or not str(department_id).strip():
            raise InputValidationError("Missing or empty mandatory field: 'department_id'")
        department_id_str = str(department_id).strip()

        # 5. Optional Context Fields with Training Pipeline Defaults
        vendor_rating_val = payload.get("vendor_rating", 3.5)
        if vendor_rating_val is None:
            vendor_rating_val = 3.5
        try:
            vendor_rating = float(vendor_rating_val)
            if not math.isfinite(vendor_rating):
                raise ValueError()
        except (ValueError, TypeError):
            raise InputValidationError("Field 'vendor_rating' must be a finite numeric value.")

        payment_terms_val = payload.get("payment_terms_days", 30)
        if payment_terms_val is None:
            payment_terms_val = 30
        try:
            payment_terms_days = int(payment_terms_val)
            if payment_terms_days < 0:
                raise ValueError()
        except (ValueError, TypeError):
            raise InputValidationError("Field 'payment_terms_days' must be a non-negative integer.")

        budget_val = payload.get("annual_budget_baseline", 10000000.0)
        if budget_val is None:
            budget_val = 10000000.0
        try:
            annual_budget_baseline = float(budget_val)
            if not math.isfinite(annual_budget_baseline) or annual_budget_baseline < 0:
                raise ValueError()
        except (ValueError, TypeError):
            raise InputValidationError("Field 'annual_budget_baseline' must be a finite non-negative number.")

        vendor_category = str(payload.get("vendor_category") or "General Supplier").strip()
        division = str(payload.get("division") or "General Operations").strip()
        payment_method = str(payload.get("payment_method") or "OTHER").strip()
        transaction_type = str(payload.get("transaction_type") or "PURCHASE").strip()
        employment_type = str(payload.get("employment_type") or "FULL_TIME").strip()

        # is_preferred_vendor string conversion ('1' or '0')
        pref_raw = payload.get("is_preferred_vendor", False)
        if isinstance(pref_raw, str):
            is_pref_str = "1" if pref_raw.strip().lower() in ("true", "1", "yes") else "0"
        else:
            is_pref_str = "1" if bool(pref_raw) else "0"

        # 6. Construct exact 1-row DataFrame for pipeline
        feature_dict = {
            "description": [description],
            "amount": [amount],
            "log_amount": [log_amount],
            "vendor_rating": [vendor_rating],
            "payment_terms_days": [payment_terms_days],
            "annual_budget_baseline": [annual_budget_baseline],
            "month": [month],
            "day": [day],
            "quarter": [quarter],
            "vendor_category": [vendor_category],
            "division": [division],
            "department_id": [department_id_str],
            "payment_method": [payment_method],
            "transaction_type": [transaction_type],
            "is_preferred_vendor": [is_pref_str],
            "employment_type": [employment_type],
            "day_of_week": [str(day_of_week)],
            "is_weekend": [str(is_weekend)]
        }

        return pd.DataFrame(feature_dict)

    def classify_expense(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes expense classification inference.

        Returns structured result conforming to Step 7.4B-2A response contract.
        """
        start_time = time.perf_counter()

        # Validate and build input features
        X = self._validate_and_prepare_features(payload)

        # Execute model prediction
        predicted_category = str(self._model.predict(X)[0])
        probabilities = self._model.predict_proba(X)[0]

        pred_idx = self.classes_.index(predicted_category)
        raw_score = float(probabilities[pred_idx])

        # Compute top alternative categories (excluding top suggested category)
        sorted_indices = np.argsort(probabilities)[::-1]
        alternatives: List[Dict[str, Any]] = []
        for idx in sorted_indices:
            cls_name = self.classes_[idx]
            if cls_name != predicted_category:
                alternatives.append({
                    "category": cls_name,
                    "raw_decision_score": round(float(probabilities[idx]), 4)
                })
            if len(alternatives) >= 3:
                break

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Sanitized audit logging (Never logging descriptions or PII)
        logger.info(
            "Expense classified successfully: predicted_category='%s', raw_score=%.4f, latency=%.2fms",
            predicted_category,
            raw_score,
            elapsed_ms
        )

        return {
            "status": "SUCCESS",
            "suggested_category": predicted_category,
            "raw_decision_score": round(raw_score, 4),
            "score_interpretation": "RAW_UNCALIBRATED_MODEL_SCORE",
            "requires_manual_confirmation": True,
            "alternatives": alternatives,
            "inference_status": "EVALUATED",
            "provenance": {
                "model_version": "2.0.0",
                "model_name": "LogisticRegression",
                "artifact_sha256": EXPECTED_SHA256,
                "data_provenance": "SYNTHETIC_V2_DATASET",
                "inference_latency_ms": elapsed_ms
            }
        }
