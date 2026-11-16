"""
STS Fintech – Milestone 2 – Step 7.2D: Comprehensive Test Suite
Validates Budget Prediction (Model D) and Vendor Analysis (Model E).
Verifies aggregation, reconciliation, temporal leakage prevention,
data quality handling, reproducibility, and error handling.
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

from ml.training.train_budget_predictor import (
    aggregate_monthly_budgets,
    engineer_budget_features,
    audit_budget_transaction_reconciliation,
    classify_risk_tier,
    FEATURE_COLS,
    EXCLUDED_LEAKAGE_COLS
)
from ml.training.train_vendor_analyzer import (
    audit_vendor_relationships,
    aggregate_vendor_metrics,
    compute_vendor_risk_score,
    SCORING_WEIGHTS
)
from ml.prediction.budget_predictor import BudgetPredictor
from ml.prediction.vendor_analyzer import VendorAnalyzer


BUDGETS_PATH = "data/raw/budgets.csv"
TRANSACTIONS_PATH = "data/raw/transactions.csv"
VENDORS_PATH = "data/raw/vendors.csv"
INVOICES_PATH = "data/raw/invoices.csv"
BILLS_PATH = "data/raw/bills.csv"
BUDGET_MODEL_PATH = "ml/models/budget_predictor.joblib"
VENDOR_MODEL_PATH = "ml/models/vendor_analyzer.joblib"
METADATA_PATH = "ml/models/step_7_2d_metadata.json"


# ==============================================================================
# 1. BUDGET PREDICTION TESTS
# ==============================================================================

def test_preserve_original_allocation_records():
    """Verify raw budgets.csv remains untouched with exactly 5,000 records."""
    df = pd.read_csv(BUDGETS_PATH)
    assert len(df) == 5000, f"Expected 5,000 raw budget records, found {len(df)}"
    assert set(df.columns) == {
        "budget_id", "department_id", "budget_year", "budget_month",
        "allocated_amount", "actual_amount", "remaining_amount", "utilization_percentage"
    }
    assert (df["allocated_amount"] > 0).all(), "Allocated amount contains non-positive values"
    assert (df["actual_amount"] > 0).all(), "Actual amount contains non-positive values"


def test_budget_monthly_aggregation():
    """Verify allocation line aggregation preserves department-month consistency."""
    df = pd.read_csv(BUDGETS_PATH)
    agg = aggregate_monthly_budgets(df)

    assert len(agg) == 1179, f"Expected 1,179 aggregated department-month rows, got {len(agg)}"
    assert agg["department_id"].nunique() == 50, "Expected all 50 departments present"
    assert agg["line_count"].sum() == 5000, "Sum of line counts must equal 5,000"
    assert (agg["utilization_percentage"] > 0).all(), "All utilization percentages must be > 0"

    # Mathematical consistency
    calc_util = (agg["actual_amount"] / agg["allocated_amount"]) * 100.0
    np.testing.assert_allclose(agg["utilization_percentage"].values, calc_util.values, rtol=1e-5)


def test_budget_transaction_reconciliation_audit():
    """Verify reconciliation calculates correlation and scale discrepancy properly."""
    b_df = pd.read_csv(BUDGETS_PATH)
    t_df = pd.read_csv(TRANSACTIONS_PATH)

    audit = audit_budget_transaction_reconciliation(b_df, t_df)
    assert audit["matched_department_months"] > 1000
    assert audit["reconciled"] is False, "Synthetic budgets and transactions must not be marked reconciled"
    assert audit["pearson_correlation"] < 0.1, "Expected zero or near-zero correlation"
    assert audit["median_budget_to_tx_ratio"] > 100.0, "Expected massive scale difference (~380x)"


def test_temporal_leakage_prevention():
    """Ensure no future information or target columns leak into feature set."""
    b_df = pd.read_csv(BUDGETS_PATH)
    agg = aggregate_monthly_budgets(b_df)
    feat = engineer_budget_features(agg)

    for col in EXCLUDED_LEAKAGE_COLS:
        assert col not in FEATURE_COLS, f"Target column '{col}' found in FEATURE_COLS!"

    # Ensure lag_1 is indeed from strictly prior period (warmup dropped index 0, so loc[0] is month 2 whose lag_1 is month 1)
    dept1_agg = agg[agg["department_id"] == 1].sort_values("period").reset_index(drop=True)
    sample_dept = feat[feat["department_id"] == 1].sort_values("period").reset_index(drop=True)
    if len(sample_dept) >= 2:
        assert sample_dept.loc[0, "lag_1_util"] == dept1_agg.loc[0, "utilization_percentage"]
        assert sample_dept.loc[1, "lag_1_util"] == dept1_agg.loc[1, "utilization_percentage"]


def test_budget_model_loading_and_prediction():
    """Verify BudgetPredictor loads artifact and produces expected outputs."""
    predictor = BudgetPredictor(BUDGET_MODEL_PATH)
    res = predictor.predict(
        department_id=1,
        budget_year=2026,
        budget_month=11,
        allocated_amount=10000000.0
    )

    assert "expected_final_utilization_pct" in res
    assert "expected_final_expenditure" in res
    assert "estimated_remaining_budget" in res
    assert "predicted_overspending_risk_tier" in res
    assert res["predicted_overspending_risk_tier"] in [
        "Within Budget", "Approaching Budget Limit", "Potential Budget Overrun"
    ]
    assert 0.0 <= res["overspending_risk_probability"] <= 1.0


def test_budget_snapshot_runrate_feature():
    """Verify intra-month snapshot calculates elapsed pace correctly without leaking."""
    predictor = BudgetPredictor(BUDGET_MODEL_PATH)
    res = predictor.predict(
        department_id=2,
        budget_year=2026,
        budget_month=11,
        allocated_amount=10000000.0,
        month_to_date_spend=2500000.0,
        as_of_day=15
    )
    assert "intra_month_snapshot" in res
    snap = res["intra_month_snapshot"]
    assert snap["as_of_day"] == 15
    assert snap["days_in_month"] == 30
    assert snap["elapsed_month_pct"] == 50.0
    assert snap["projected_mtd_full_month_runrate"] == 5000000.0
    assert snap["mtd_utilization_pace_pct"] == 50.0


def test_budget_reproducibility():
    """Verify deterministic outputs for identical inputs."""
    p1 = BudgetPredictor(BUDGET_MODEL_PATH)
    p2 = BudgetPredictor(BUDGET_MODEL_PATH)

    r1 = p1.predict(department_id=5, budget_year=2026, budget_month=10, allocated_amount=8500000.0)
    r2 = p2.predict(department_id=5, budget_year=2026, budget_month=10, allocated_amount=8500000.0)

    assert r1["expected_final_utilization_pct"] == r2["expected_final_utilization_pct"]
    assert r1["expected_final_expenditure"] == r2["expected_final_expenditure"]


def test_budget_invalid_input_handling():
    """Verify rejection of invalid budget inputs."""
    predictor = BudgetPredictor(BUDGET_MODEL_PATH)

    with pytest.raises(ValueError):
        predictor.predict(department_id=1, budget_year=2026, budget_month=11, allocated_amount=-500.0)

    with pytest.raises(ValueError):
        predictor.predict(department_id=1, budget_year=2026, budget_month=13, allocated_amount=1000000.0)

    with pytest.raises(ValueError):
        predictor.predict(department_id=999, budget_year=2026, budget_month=5, allocated_amount=1000000.0)


# ==============================================================================
# 2. VENDOR ANALYSIS TESTS
# ==============================================================================

def test_vendor_relationship_audit():
    """Verify foreign key integrity across vendors, invoices, bills, and transactions."""
    audit = audit_vendor_relationships()
    assert audit["vendor_count"] == 2000
    assert audit["invoice_count"] == 5000
    assert audit["bill_count"] == 5000
    assert audit["transaction_count"] == 10000

    assert audit["foreign_key_integrity"]["invoices_orphaned_vendors"] == 0
    assert audit["foreign_key_integrity"]["bills_orphaned_vendors"] == 0
    assert audit["foreign_key_integrity"]["transactions_orphaned_vendors"] == 0
    assert audit["identifier_uniqueness"]["invoices_unique"] is True
    assert audit["identifier_uniqueness"]["bills_unique"] is True


def test_vendor_missing_payment_date_handled():
    """Verify 100% missing payment_date in bills.csv is excluded."""
    bills = pd.read_csv(BILLS_PATH)
    assert bills["payment_date"].isna().sum() == len(bills), "payment_date must be 100% null in bills.csv"

    audit = audit_vendor_relationships()
    assert audit["bills_missing_payment_date"]["missing_percentage"] == 100.0
    assert "Excluded" in audit["bills_missing_payment_date"]["handling"]


def test_vendor_risk_leakage_and_rule_scoring():
    """Verify multi-factor rule scoring operates without target leakage."""
    analyzer = VendorAnalyzer(VENDOR_MODEL_PATH)
    v1 = analyzer.analyze_vendor("VEN00001")

    assert "risk_evaluation" in v1
    eval_res = v1["risk_evaluation"]
    assert 0.0 <= eval_res["risk_score"] <= 100.0
    assert eval_res["risk_tier"] in ["LOW", "MEDIUM", "HIGH – REQUIRES VERIFICATION"]

    breakdown = eval_res["factor_breakdown"]
    assert set(breakdown.keys()) == {
        "invoice_rejection_score",
        "duplicate_invoicing_score",
        "overdue_payment_score",
        "transaction_anomaly_score",
        "vendor_rating_deficit_score"
    }


def test_vendor_analyzer_high_risk_flagging_disclaimer():
    """Verify high risk vendors are flagged for verification and compliance disclaimer exists."""
    analyzer = VendorAnalyzer(VENDOR_MODEL_PATH)
    high_risk = analyzer.get_high_risk_vendors()

    assert len(high_risk) > 0, "Expected at least one vendor flagged as HIGH risk"
    top_v = analyzer.analyze_vendor(high_risk[0]["vendor_id"])

    assert top_v["risk_evaluation"]["risk_tier"] == "HIGH – REQUIRES VERIFICATION"
    assert "verification" in top_v["recommended_action"].lower()
    assert "Does not constitute a determination of fraud" in top_v["compliance_disclaimer"]


def test_vendor_invalid_input_handling():
    """Verify rejection of unknown or invalid vendor IDs."""
    analyzer = VendorAnalyzer(VENDOR_MODEL_PATH)

    with pytest.raises(ValueError):
        analyzer.analyze_vendor("")

    with pytest.raises(KeyError):
        analyzer.analyze_vendor("VEN99999")


# ==============================================================================
# 3. METADATA AND ARTIFACT PERSISTENCE
# ==============================================================================

def test_step_7_2d_artifacts_and_checksums():
    """Verify presence and valid SHA-256 hashes of all Step 7.2D artifacts."""
    assert os.path.exists(BUDGET_MODEL_PATH), f"Missing {BUDGET_MODEL_PATH}"
    assert os.path.exists(VENDOR_MODEL_PATH), f"Missing {VENDOR_MODEL_PATH}"
    assert os.path.exists(METADATA_PATH), f"Missing {METADATA_PATH}"

    with open(METADATA_PATH) as f:
        meta = json.load(f)

    # Compute live checksums
    def get_hash(p):
        h = hashlib.sha256()
        with open(p, "rb") as fp:
            while chunk := fp.read(65536):
                h.update(chunk)
        return h.hexdigest()

    b_hash = get_hash(BUDGET_MODEL_PATH)
    v_hash = get_hash(VENDOR_MODEL_PATH)

    assert meta["model_d_budget_prediction"]["sha256_checksum"] == b_hash
    assert meta["model_e_vendor_analysis"]["sha256_checksum"] == v_hash
