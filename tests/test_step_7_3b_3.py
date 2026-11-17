"""
STS FINTECH AI FINANCIAL INTELLIGENCE
Automated Test Suite for Milestone 2 - Step 7.3B-3
Model D: V2 Budget Utilization Prediction
Model E: V2 Vendor Intelligence & Accounts Payable Risk Suite

Tests cover:
1. Dataset Integrity (Snapshots, Invoices, Bills, Vendors)
2. Target Leakage Elimination in Budget & Vendor Features
3. Model D Budget Predictor Inference, Bounds, and Advisory Risk Tiers
4. Model D Monotonic Error Reduction across Snapshot Horizons (Day 7, 14, 21)
5. Model E Sub-engine 1: Duplicate Matcher Deterministic Precision & Leakage Safeguards
6. Model E Sub-engine 2: Late Payment Prediction Probabilities & Contractual Features
7. Model E Sub-engine 3: Multi-Factor Vendor Risk Scoring Bounds (0-100) and Tiers
8. Metadata and SHA-256 Checksum Verification
9. V1 & Prior V2 Model Artifact Preservation
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
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.prediction.budget_predictor_v2 import BudgetPredictorV2
from ml.prediction.vendor_analyzer_v2 import VendorAnalyzerV2, OperationalDuplicateMatcher

V2_MODELS_DIR = os.path.join(BASE_DIR, "ml", "models", "v2")
V1_MODELS_DIR = os.path.join(BASE_DIR, "ml", "models")
DATA_V2_DIR = os.path.join(BASE_DIR, "data", "v2", "raw")
METADATA_PATH = os.path.join(V2_MODELS_DIR, "step_7_3b_3_metadata.json")

BUDGET_MODEL_PATH = os.path.join(V2_MODELS_DIR, "budget_predictor_v2.joblib")
DUP_MODEL_PATH = os.path.join(V2_MODELS_DIR, "vendor_duplicate_detector_v2.joblib")
PAY_MODEL_PATH = os.path.join(V2_MODELS_DIR, "vendor_payment_predictor_v2.joblib")
RISK_MODEL_PATH = os.path.join(V2_MODELS_DIR, "vendor_screening_v2.joblib")


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class TestStep73B3DatasetIntegrity:
    """Verifies V2 raw dataset files and row counts for budget and AP ledgers."""

    def test_budget_snapshots_count_and_columns(self):
        df = pd.read_csv(os.path.join(DATA_V2_DIR, "budget_snapshots.csv"))
        assert len(df) == 5400, f"Expected exactly 5,400 snapshots, got {len(df)}"
        assert set(df["as_of_day"].unique()) == {7, 14, 21}
        assert len(df["department_id"].unique()) == 50
        assert "final_actual_amount" in df.columns
        assert "final_utilization_pct" in df.columns
        assert "is_overspent_final" in df.columns

    def test_invoices_count_and_columns(self):
        df = pd.read_csv(os.path.join(DATA_V2_DIR, "invoices.csv"))
        assert len(df) == 10000, f"Expected 10,000 invoices, got {len(df)}"
        assert "is_duplicate" in df.columns
        assert df["is_duplicate"].sum() == 270

    def test_bills_count_and_columns(self):
        df = pd.read_csv(os.path.join(DATA_V2_DIR, "bills.csv"))
        assert len(df) == 10000, f"Expected 10,000 bills, got {len(df)}"
        assert "is_late_payment" in df.columns
        assert 0.18 <= df["is_late_payment"].mean() <= 0.25

    def test_vendors_count(self):
        df = pd.read_csv(os.path.join(DATA_V2_DIR, "vendors.csv"))
        assert len(df) == 2000, f"Expected 2,000 vendors, got {len(df)}"


class TestModelDBudgetPredictor:
    """Tests Model D: Budget Utilization Predictor."""

    @pytest.fixture(scope="class")
    def predictor(self):
        return BudgetPredictorV2(BUDGET_MODEL_PATH)

    def test_no_target_leakage_in_features(self, predictor):
        leakage_cols = ["final_actual_amount", "final_utilization_pct", "is_overspent_final", "actual_amount"]
        for col in leakage_cols:
            assert col not in predictor.feature_names, f"Target leakage detected in features: {col}"

    def test_input_validation(self, predictor):
        with pytest.raises(ValueError, match="allocated_amount must be strictly positive"):
            predictor.predict(department_id=1, budget_year=2026, budget_month=9, as_of_day=7, allocated_amount=0, month_to_date_actual=100)

        with pytest.raises(ValueError, match="allocated_amount must be strictly positive"):
            predictor.predict(department_id=1, budget_year=2026, budget_month=9, as_of_day=7, allocated_amount=-500, month_to_date_actual=100)

        with pytest.raises(ValueError, match="month_to_date_actual must be non-negative"):
            predictor.predict(department_id=1, budget_year=2026, budget_month=9, as_of_day=7, allocated_amount=10000, month_to_date_actual=-50)

        with pytest.raises(ValueError, match="budget_month must be between 1 and 12"):
            predictor.predict(department_id=1, budget_year=2026, budget_month=13, as_of_day=7, allocated_amount=10000, month_to_date_actual=100)

        with pytest.raises(ValueError, match="as_of_day must be between 1 and"):
            predictor.predict(department_id=1, budget_year=2026, budget_month=9, as_of_day=32, allocated_amount=10000, month_to_date_actual=100)

    def test_prediction_output_structure_and_bounds(self, predictor):
        res = predictor.predict(
            department_id=5,
            budget_year=2026,
            budget_month=9,
            as_of_day=14,
            allocated_amount=2000000.0,
            month_to_date_actual=800000.0
        )
        assert res["predicted_utilization_pct"] >= 0.0
        assert res["predicted_final_actual"] >= 0.0
        assert res["predicted_final_actual"] >= res["month_to_date_actual"]
        assert 0.0 <= res["overrun_probability"] <= 1.0
        assert res["advisory_risk_tier"] in [
            "Within Budget",
            "Approaching Budget Limit",
            "Potential Budget Overrun – Requires Verification"
        ]
        assert "recommended_action" in res
        assert "baselines" in res
        assert "linear_runrate_actual" in res["baselines"]

    def test_overrun_risk_tier_assignment(self, predictor):
        # Scenario 1: Low spend -> Within Budget
        res_low = predictor.predict(
            department_id=1,
            budget_year=2026,
            budget_month=9,
            as_of_day=21,
            allocated_amount=5000000.0,
            month_to_date_actual=1000000.0
        )
        assert res_low["advisory_risk_tier"] == "Within Budget"
        assert res_low["potential_overrun"] is False

        # Scenario 2: Severe overspend -> Potential Budget Overrun
        res_high = predictor.predict(
            department_id=1,
            budget_year=2026,
            budget_month=9,
            as_of_day=21,
            allocated_amount=2000000.0,
            month_to_date_actual=2500000.0
        )
        assert res_high["advisory_risk_tier"] == "Potential Budget Overrun – Requires Verification"
        assert res_high["potential_overrun"] is True
        assert res_high["overrun_probability"] > 0.80

    def test_snapshot_day_monotonic_improvement(self, predictor):
        """Verifies that later snapshots (Day 21) have lower MAE than early snapshots (Day 7)."""
        bundle = joblib.load(BUDGET_MODEL_PATH)
        day_m = bundle["day_metrics"]
        mae_7 = day_m["day_7"]["model_util_mae"]
        mae_14 = day_m["day_14"]["model_util_mae"]
        mae_21 = day_m["day_21"]["model_util_mae"]
        assert mae_21 < mae_14 <= mae_7, f"Expected monotonic MAE reduction: Day 7 ({mae_7}) > Day 14 ({mae_14}) > Day 21 ({mae_21})"


class TestModelEVendorIntelligence:
    """Tests Model E: Vendor Intelligence Suite."""

    @pytest.fixture(scope="class")
    def analyzer(self):
        return VendorAnalyzerV2(RISK_MODEL_PATH, DUP_MODEL_PATH, PAY_MODEL_PATH)

    def test_duplicate_screening_leakage_exclusion(self):
        bundle = joblib.load(DUP_MODEL_PATH)
        feature_names = bundle["feature_names"]
        assert "rejection_reason" not in feature_names
        assert "invoice_status" not in feature_names

    def test_operational_duplicate_matcher(self, analyzer):
        matcher = analyzer.matcher
        assert matcher is not None

        # Clean check (non-duplicate)
        res_clean = analyzer.screen_invoice_duplicate("VEN00001", 9999999.99, "2026-05-15")
        assert res_clean["is_suspected_duplicate"] is False
        assert res_clean["match_count"] == 0

        # Exact match injection
        matcher.registry.append({
            "invoice_id": "TEST_INV_01",
            "vendor_id": "VEN_TEST",
            "total_amount": 54321.00,
            "invoice_date": pd.to_datetime("2026-06-01")
        })

        # Test exact match within 10 days
        res_dup = matcher.check_duplicate("VEN_TEST", 54321.00, "2026-06-10")
        assert res_dup["is_suspected_duplicate"] is True
        assert res_dup["match_count"] >= 1
        assert res_dup["matches"][0]["matched_invoice_id"] == "TEST_INV_01"
        assert res_dup["matches"][0]["match_type"] == "EXACT"

        # Test near match within 1%
        res_near = matcher.check_duplicate("VEN_TEST", 54350.00, "2026-06-05")
        assert res_near["is_suspected_duplicate"] is True
        assert res_near["matches"][0]["match_type"] == "NEAR_MATCH"

        # Test outside time window (45 days)
        res_far = matcher.check_duplicate("VEN_TEST", 54321.00, "2026-08-01")
        assert res_far["is_suspected_duplicate"] is False

    def test_late_payment_leakage_exclusion(self):
        bundle = joblib.load(PAY_MODEL_PATH)
        features = bundle["feature_names"]
        assert "payment_status" not in features
        assert "payment_date" not in features

    def test_late_payment_prediction_bounds(self, analyzer):
        res = analyzer.predict_late_payment(
            vendor_id="VEN00001",
            total_amount=100000.0,
            bill_date="2026-09-01",
            due_date="2026-09-30"
        )
        assert 0.0 <= res["late_payment_probability"] <= 1.0
        assert isinstance(res["is_suspected_late_payment"], bool)
        assert res["terms_days"] == 29
        assert "risk_advisory" in res

    def test_vendor_risk_screening_all_profiles(self, analyzer):
        assert len(analyzer.profiles) == 2000

        for vid, p in analyzer.profiles.items():
            assert 0.0 <= p["risk_score"] <= 100.0
            assert p["risk_tier"] in ["LOW", "MEDIUM", "HIGH – REQUIRES VERIFICATION"]
            factors = p["factor_scores"]
            for f_name, f_val in factors.items():
                assert 0.0 <= f_val <= 100.0

    def test_high_risk_vendors_retrieval(self, analyzer):
        high_risk = analyzer.get_high_risk_vendors()
        assert len(high_risk) > 0
        for v in high_risk:
            assert v["risk_tier"] == "HIGH – REQUIRES VERIFICATION"
            assert v["risk_score"] > 50.0

    def test_analyze_vendor_input_validation(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.analyze_vendor("")
        with pytest.raises(KeyError):
            analyzer.analyze_vendor("UNKNOWN_VENDOR_9999")


class TestArtifactIntegrityAndPreservation:
    """Verifies all model artifacts, SHA-256 signatures, and V1 preservation."""

    def test_all_step_7_3b_3_artifacts_exist_and_sizes(self):
        artifacts = [BUDGET_MODEL_PATH, DUP_MODEL_PATH, PAY_MODEL_PATH, RISK_MODEL_PATH]
        for p in artifacts:
            assert os.path.exists(p), f"Artifact missing: {p}"
            assert os.path.getsize(p) > 100_000, f"Artifact too small: {p}"

    def test_step_7_3b_3_metadata_sha256_match(self):
        assert os.path.exists(METADATA_PATH), f"Missing metadata file: {METADATA_PATH}"
        with open(METADATA_PATH, "r") as f:
            meta = json.load(f)

        assert meta["step"] == "7.3B-3"
        artifacts = meta["artifacts"]

        for name, info in artifacts.items():
            file_path = os.path.join(BASE_DIR, info["file"])
            actual_sha = compute_sha256(file_path)
            assert actual_sha == info["sha256"], f"SHA256 mismatch for {name}: expected {info['sha256']}, got {actual_sha}"

    def test_v1_and_prior_v2_artifacts_preserved(self):
        v1_artifacts = [
            "expense_classifier.joblib",
            "anomaly_detector.joblib",
            "expense_forecaster.joblib",
            "budget_predictor.joblib",
            "vendor_analyzer.joblib"
        ]
        for f in v1_artifacts:
            path = os.path.join(V1_MODELS_DIR, f)
            assert os.path.exists(path), f"V1 artifact missing or deleted: {path}"

        prior_v2_artifacts = [
            "expense_classifier_v2.joblib",
            "anomaly_detector_v2.joblib",
            "expense_forecaster_v2.joblib"
        ]
        for f in prior_v2_artifacts:
            path = os.path.join(V2_MODELS_DIR, f)
            assert os.path.exists(path), f"Prior V2 artifact missing or deleted: {path}"
