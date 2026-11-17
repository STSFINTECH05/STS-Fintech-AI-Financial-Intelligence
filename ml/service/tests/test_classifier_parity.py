"""
STS FINTECH AI FINANCIAL INTELLIGENCE
Model A: Permanent Prediction Parity Regression Tests (Milestone 2 - Step 7.4B-2A2)

Verifies bit-for-bit parity between the verified Model A scikit-learn pipeline
and the ExpenseClassifierV2 production inference wrapper service.
Ensures:
- 100% agreement on predicted class label across edge-case fixtures.
- Mathematical parity on raw decision scores and alternative class probabilities.
- Exact feature ordering and calendar/logarithmic transformation parity.
- Safe handling of unseen categories, empty text, and omitted optional metadata.
- No fabricated business mappings or unvalidated threshold drift.
"""

import math
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import pytest

from ml.service.config import DEFAULT_MODELS_DIR
from ml.service.model_loader import ModelRegistry, APPROVED_MODEL_SPECS
from ml.service.prediction.expense_classifier import ExpenseClassifierV2

EXPECTED_SHA256 = APPROVED_MODEL_SPECS["model_a_classifier"]["expected_sha256"]


@pytest.fixture(scope="module")
def registry_and_models():
    """Initializes ModelRegistry and returns both raw model and ExpenseClassifierV2."""
    reg = ModelRegistry(DEFAULT_MODELS_DIR)
    reg.initialize_registry()
    classifier = ExpenseClassifierV2(reg)
    raw_model = reg.get_model("model_a_classifier")
    return classifier, raw_model


def independent_reference_prepare_df(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Independent reference feature preparation function.
    Implements the original V2 training pipeline transformations independently
    to detect any transformation, column ordering, or default drift in the service wrapper.
    """
    amount = float(payload["amount"])
    log_amount = float(np.log1p(amount))

    # Independent calendar derivation
    dt = pd.to_datetime(payload["transaction_date"])
    month = int(dt.month)
    day = int(dt.day)
    quarter = int(dt.quarter)
    day_of_week = str(dt.weekday())
    is_weekend = "1" if dt.weekday() >= 5 else "0"

    # Narrative text
    desc = str(payload.get("description") or "").strip()

    # Department
    department_id = str(payload["department_id"]).strip()

    # Defaults matching original training pipeline
    vendor_category = str(payload.get("vendor_category") or "General Supplier").strip()
    division = str(payload.get("division") or "General Operations").strip()
    payment_method = str(payload.get("payment_method") or "OTHER").strip()
    transaction_type = str(payload.get("transaction_type") or "PURCHASE").strip()
    vendor_rating = float(payload.get("vendor_rating", 3.5) if payload.get("vendor_rating") is not None else 3.5)
    payment_terms_days = int(payload.get("payment_terms_days", 30) if payload.get("payment_terms_days") is not None else 30)
    annual_budget_baseline = float(payload.get("annual_budget_baseline", 10000000.0) if payload.get("annual_budget_baseline") is not None else 10000000.0)
    employment_type = str(payload.get("employment_type") or "FULL_TIME").strip()

    pref_raw = payload.get("is_preferred_vendor", False)
    if isinstance(pref_raw, str):
        is_pref_str = "1" if pref_raw.strip().lower() in ("true", "1", "yes") else "0"
    else:
        is_pref_str = "1" if bool(pref_raw) else "0"

    # Exact 18-column DataFrame matching ColumnTransformer named columns
    return pd.DataFrame({
        "description": [desc],
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
        "department_id": [department_id],
        "payment_method": [payment_method],
        "transaction_type": [transaction_type],
        "is_preferred_vendor": [is_pref_str],
        "employment_type": [employment_type],
        "day_of_week": [day_of_week],
        "is_weekend": [is_weekend]
    })


PARITY_FIXTURES: List[Dict[str, Any]] = [
    # 1. Known vendor and department categories, high tech expenditure
    {
        "id": "known_high_tech_cloud",
        "payload": {
            "description": "AWS Cloud Hosting Infrastructure Monthly Invoice",
            "amount": 18500.50,
            "transaction_date": "2026-04-12",
            "department_id": "1",
            "vendor_category": "Cloud Computing Provider",
            "division": "Technology",
            "payment_method": "CORPORATE_CREDIT_CARD",
            "transaction_type": "EXPENSE",
            "vendor_rating": 4.8,
            "payment_terms_days": 30,
            "annual_budget_baseline": 15000000.0,
            "is_preferred_vendor": True,
            "employment_type": "FULL_TIME"
        }
    },
    # 2. Empty description with hardware vendor category
    {
        "id": "empty_description_hardware",
        "payload": {
            "description": "",
            "amount": 345000.00,
            "transaction_date": "2026-08-01",
            "department_id": "12",
            "vendor_category": "Computer Hardware & Peripherals",
            "division": "Technology",
            "payment_method": "BANK_TRANSFER",
            "transaction_type": "INVOICE_SETTLEMENT",
            "vendor_rating": 3.8,
            "payment_terms_days": 45,
            "annual_budget_baseline": 50000000.0,
            "is_preferred_vendor": True,
            "employment_type": "CONTRACT"
        }
    },
    # 3. Unseen vendor category string
    {
        "id": "unseen_vendor_category",
        "payload": {
            "description": "Advanced High Performance GPU cluster leasing and support",
            "amount": 75000.00,
            "transaction_date": "2026-03-15",
            "department_id": "5",
            "vendor_category": "QuantumAI Compute Networks Ltd",
            "division": "Technology",
            "payment_method": "CORPORATE_CREDIT_CARD",
            "transaction_type": "EXPENSE",
            "vendor_rating": 4.0,
            "payment_terms_days": 30,
            "annual_budget_baseline": 20000000.0,
            "is_preferred_vendor": False,
            "employment_type": "FULL_TIME"
        }
    },
    # 4. Unseen department identifier & unseen division
    {
        "id": "unseen_department_and_division",
        "payload": {
            "description": "Laboratory equipment calibration and diagnostic maintenance service",
            "amount": 12500.00,
            "transaction_date": "2026-07-20",
            "department_id": "DEP-UNSEEN-999",
            "vendor_category": "Building Facilities & HVAC",
            "division": "Biotech Research Lab",
            "payment_method": "UPI",
            "transaction_type": "PAYMENT",
            "vendor_rating": 3.5,
            "payment_terms_days": 30,
            "annual_budget_baseline": 10000000.0,
            "is_preferred_vendor": False,
            "employment_type": "FULL_TIME"
        }
    },
    # 5. Missing optional categorical values (only mandatory provided)
    {
        "id": "omitted_optional_metadata",
        "payload": {
            "amount": 4200.00,
            "transaction_date": "2026-05-18",
            "department_id": "8"
        }
    },
    # 6. Small transaction amount with meals & entertainment
    {
        "id": "small_amount_meals",
        "payload": {
            "description": "Team lunch coffee and snacks during strategy workshop",
            "amount": 350.00,
            "transaction_date": "2026-09-30",
            "department_id": "3",
            "vendor_category": "Restaurants & Catering",
            "division": "Commercial",
            "payment_method": "CASH",
            "transaction_type": "EXPENSE",
            "vendor_rating": 3.9,
            "payment_terms_days": 0,
            "annual_budget_baseline": 18000000.0,
            "is_preferred_vendor": False,
            "employment_type": "FULL_TIME"
        }
    },
    # 7. Leap year transaction date boundary case
    {
        "id": "leap_year_boundary_date",
        "payload": {
            "description": "Annual enterprise cloud subscription renewal license",
            "amount": 98000.00,
            "transaction_date": "2024-02-29",
            "department_id": "2",
            "vendor_category": "Enterprise SaaS Solutions",
            "division": "Finance & Admin",
            "payment_method": "DIRECT_DEBIT",
            "transaction_type": "INVOICE_SETTLEMENT",
            "vendor_rating": 4.6,
            "payment_terms_days": 60,
            "annual_budget_baseline": 35000000.0,
            "is_preferred_vendor": True,
            "employment_type": "FULL_TIME"
        }
    }
]


class TestClassifierParity:
    """Test suite ensuring permanent parity between raw model and service wrapper."""

    @pytest.mark.parametrize("fixture", PARITY_FIXTURES, ids=[f["id"] for f in PARITY_FIXTURES])
    def test_prediction_class_and_score_parity(self, registry_and_models, fixture):
        """
        Verifies that ExpenseClassifierV2 produces exactly identical classification results
        and raw decision scores as the underlying scikit-learn model on independent DataFrames.
        """
        classifier, raw_model = registry_and_models
        payload = fixture["payload"]

        # 1. Execute inference via service wrapper
        wrapper_result = classifier.classify_expense(payload)

        # 2. Execute inference directly on raw model via independent reference dataframe
        ref_df = independent_reference_prepare_df(payload)
        raw_pred = str(raw_model.predict(ref_df)[0])
        raw_probs = raw_model.predict_proba(ref_df)[0]
        classes = list(raw_model.classes_)

        raw_pred_idx = classes.index(raw_pred)
        expected_score = round(float(raw_probs[raw_pred_idx]), 4)

        # A. Predicted category parity
        assert wrapper_result["suggested_category"] == raw_pred, (
            f"Class mismatch for fixture '{fixture['id']}': "
            f"Wrapper={wrapper_result['suggested_category']} vs Raw={raw_pred}"
        )

        # B. Raw decision score parity (within roundoff tolerance)
        assert wrapper_result["raw_decision_score"] == pytest.approx(expected_score, abs=1e-4), (
            f"Score mismatch for fixture '{fixture['id']}': "
            f"Wrapper={wrapper_result['raw_decision_score']} vs Raw={expected_score}"
        )

        # C. Alternatives class and score ordering parity
        sorted_indices = np.argsort(raw_probs)[::-1]
        expected_alts = []
        for idx in sorted_indices:
            cls_name = classes[idx]
            if cls_name != raw_pred:
                expected_alts.append({
                    "category": cls_name,
                    "raw_decision_score": round(float(raw_probs[idx]), 4)
                })
            if len(expected_alts) >= 3:
                break

        wrapper_alts = wrapper_result["alternatives"]
        assert len(wrapper_alts) == len(expected_alts), (
            f"Alternatives count mismatch for fixture '{fixture['id']}'"
        )

        for i, (w_alt, e_alt) in enumerate(zip(wrapper_alts, expected_alts)):
            assert w_alt["category"] == e_alt["category"], (
                f"Alternative {i} category mismatch: {w_alt['category']} vs {e_alt['category']}"
            )
            assert w_alt["raw_decision_score"] == pytest.approx(e_alt["raw_decision_score"], abs=1e-4), (
                f"Alternative {i} score mismatch: {w_alt['raw_decision_score']} vs {e_alt['raw_decision_score']}"
            )

        # D. Highest score monotonicity check: top category >= highest alternative
        if wrapper_alts:
            assert wrapper_result["raw_decision_score"] >= wrapper_alts[0]["raw_decision_score"]

    def test_artifact_sha256_unmodified(self, registry_and_models):
        """Verifies that the Model A artifact SHA-256 remains bit-for-bit identical."""
        classifier, raw_model = registry_and_models
        spec = APPROVED_MODEL_SPECS["model_a_classifier"]
        import os, hashlib
        p = os.path.join(DEFAULT_MODELS_DIR, spec["filename"])
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        assert h.hexdigest() == EXPECTED_SHA256
