"""
STS Fintech – Milestone 2 – Step 7.3A: Comprehensive Validation Test Suite for V2 Data Generator
Verifies:
- Primary key uniqueness across all 9 V2 datasets.
- Foreign key integrity across all relational entity boundaries.
- Expense category distribution and absence of description leakage.
- Anomaly scenario diversity and ground truth provenance.
- Time-series daily continuity across 36 continuous calendar months.
- 100% exact mathematical reconciliation between transactions and budgets.
- Intra-month snapshot monotonicity (Day 7 <= Day 14 <= Day 21 <= Final).
- Accounts payable payment lifecycle and payment date population.
- Deterministic reproducibility with fixed random seed.
"""

import os
import sys
import json
import hashlib
from datetime import datetime, date
import pytest
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from ml.data_generator_v2 import (
    GeneratorConfig,
    EnterpriseDataGeneratorV2,
    EXPENSE_CATEGORIES,
    VENDOR_CATEGORIES
)

V2_DIR = os.path.join(BASE_DIR, "data/v2/raw")


@pytest.fixture(scope="module")
def v2_datasets():
    """Loads all 9 V2 datasets into memory for fast test execution."""
    data = {}
    fnames = [
        "departments.csv", "employees.csv", "vendors.csv", "salaries.csv",
        "transactions.csv", "invoices.csv", "bills.csv", "budgets.csv",
        "budget_snapshots.csv"
    ]
    for fn in fnames:
        fpath = os.path.join(V2_DIR, fn)
        assert os.path.exists(fpath), f"Missing required V2 dataset: {fpath}"
        data[fn] = pd.read_csv(fpath)
    return data


# ==============================================================================
# 1. PRIMARY KEY & FOREIGN KEY INTEGRITY TESTS
# ==============================================================================

def test_primary_key_uniqueness(v2_datasets):
    """Verify primary key uniqueness across all generated datasets."""
    pks = {
        "departments.csv": "department_id",
        "employees.csv": "employee_id",
        "vendors.csv": "vendor_id",
        "salaries.csv": "salary_id",
        "transactions.csv": "transaction_id",
        "invoices.csv": "invoice_id",
        "bills.csv": "bill_id",
        "budgets.csv": "budget_id",
        "budget_snapshots.csv": "snapshot_id"
    }
    for fname, pk in pks.items():
        df = v2_datasets[fname]
        assert df[pk].nunique() == len(df), f"Duplicate primary keys found in {fname} on {pk}"


def test_foreign_key_referential_integrity(v2_datasets):
    """Verify 100% referential integrity with zero orphaned foreign keys."""
    dept_ids = set(v2_datasets["departments.csv"]["department_id"])
    emp_ids = set(v2_datasets["employees.csv"]["employee_id"])
    vendor_ids = set(v2_datasets["vendors.csv"]["vendor_id"])
    inv_ids = set(v2_datasets["invoices.csv"]["invoice_id"])

    # Employees -> Departments
    assert set(v2_datasets["employees.csv"]["department_id"]).issubset(dept_ids)

    # Transactions -> Employees, Departments, Vendors
    assert set(v2_datasets["transactions.csv"]["employee_id"]).issubset(emp_ids)
    assert set(v2_datasets["transactions.csv"]["department_id"]).issubset(dept_ids)
    assert set(v2_datasets["transactions.csv"]["vendor_id"]).issubset(vendor_ids)

    # Invoices -> Vendors, Departments
    assert set(v2_datasets["invoices.csv"]["vendor_id"]).issubset(vendor_ids)
    assert set(v2_datasets["invoices.csv"]["department_id"]).issubset(dept_ids)

    # Bills -> Invoices, Vendors, Departments
    assert set(v2_datasets["bills.csv"]["invoice_id"]).issubset(inv_ids)
    assert set(v2_datasets["bills.csv"]["vendor_id"]).issubset(vendor_ids)
    assert set(v2_datasets["bills.csv"]["department_id"]).issubset(dept_ids)

    # Budgets & Snapshots -> Departments
    assert set(v2_datasets["budgets.csv"]["department_id"]).issubset(dept_ids)
    assert set(v2_datasets["budget_snapshots.csv"]["department_id"]).issubset(dept_ids)


# ==============================================================================
# 2. CATEGORY MODELING & LEAKAGE PREVENTION TESTS (USE CASE A)
# ==============================================================================

def test_expense_category_distribution(v2_datasets):
    """Verify all 14 categories have adequate, balanced representation (> 2,000 tx each)."""
    tx = v2_datasets["transactions.csv"]
    cat_counts = tx["category"].value_counts()

    assert len(cat_counts) == 14, f"Expected 14 categories, found {len(cat_counts)}"
    for cat in EXPENSE_CATEGORIES:
        assert cat in cat_counts, f"Missing category: {cat}"
        assert cat_counts[cat] >= 2000, f"Category {cat} has insufficient volume: {cat_counts[cat]}"


def test_category_description_leakage_prevention(v2_datasets):
    """
    CRITICAL: Assert that transaction descriptions do NOT contain the target category name.
    Guarantees models cannot classify expenses by trivial string matching.
    """
    tx = v2_datasets["transactions.csv"]

    # Check that category name is not in description
    for _, row in tx.sample(n=2000, random_state=42).iterrows():
        cat = row["category"].lower()
        desc = row["description"].lower()
        # Literal category name must not appear in description
        assert cat not in desc, f"Target category '{cat}' leaked directly into description: '{desc}'"


def test_category_vendor_relationship_correlation(v2_datasets):
    """Verify strong commercial correlation between vendor category and expense category."""
    tx = v2_datasets["transactions.csv"]
    vendors = v2_datasets["vendors.csv"].set_index("vendor_id")

    tx_merged = tx.join(vendors[["vendor_category"]], on="vendor_id")
    cloud_tx = tx_merged[tx_merged["vendor_category"] == "Cloud Computing Provider"]
    cloud_services_rate = (cloud_tx["category"] == "Cloud Services").mean()

    # Must be strong correlation (> 75%) reflecting realistic commercial accounts
    assert cloud_services_rate >= 0.75, f"Expected Cloud Services rate >= 75%, got {cloud_services_rate:.2%}"


# ==============================================================================
# 3. ANOMALY DETECTION INTEGRITY TESTS (USE CASE B)
# ==============================================================================

def test_anomaly_scenarios_diversity(v2_datasets):
    """Verify presence of diverse anomaly scenarios and hard negative examples."""
    tx = v2_datasets["transactions.csv"]
    anomaly_counts = tx["anomaly_type"].value_counts().to_dict()

    expected_types = [
        "NONE",
        "UNUSUAL_LARGE_AMOUNT",
        "SPLIT_APPROVAL_THRESHOLD",
        "OFF_HOURS_WEEKEND_OUTLIER",
        "UNRATED_OR_HIGH_RISK_VENDOR",
        "HARD_NEGATIVE_APPROVED_SURGE"
    ]
    for atype in expected_types:
        assert atype in anomaly_counts, f"Missing anomaly scenario type: {atype}"
        assert anomaly_counts[atype] > 0

    # Hard negatives must have is_anomaly = False
    hard_negatives = tx[tx["anomaly_type"] == "HARD_NEGATIVE_APPROVED_SURGE"]
    assert (hard_negatives["is_anomaly"] == False).all(), "Hard negatives must not be labeled anomalous!"

    # Overall anomaly percentage between 1.5% and 4.0%
    anomaly_rate = tx["is_anomaly"].mean() * 100.0
    assert 1.5 <= anomaly_rate <= 4.0, f"Anomaly rate out of expected range: {anomaly_rate:.2f}%"


# ==============================================================================
# 4. TIME-SERIES CONTINUITY TESTS (USE CASE C)
# ==============================================================================

def test_time_series_daily_continuity(v2_datasets):
    """Verify all 1,096 continuous calendar days from 2024-01-01 to 2026-12-31 exist."""
    tx = v2_datasets["transactions.csv"]
    unique_dates = set(tx["transaction_date"])

    start_date = date(2024, 1, 1)
    end_date = date(2026, 12, 31)
    expected_days = (end_date - start_date).days + 1

    assert len(unique_dates) == expected_days, f"Expected {expected_days} days, found {len(unique_dates)}"
    assert min(unique_dates) == "2024-01-01"
    assert max(unique_dates) == "2026-12-31"


# ==============================================================================
# 5. BUDGET RECONCILIATION & SNAPSHOT MONOTONICITY (USE CASE D)
# ==============================================================================

def test_exact_budget_transaction_reconciliation(v2_datasets):
    """
    CRITICAL: Verify exact 100% mathematical reconciliation between transactions
    and departmental monthly actual expenditures.
    """
    tx = v2_datasets["transactions.csv"].copy()
    budgets = v2_datasets["budgets.csv"].copy()

    tx["date"] = pd.to_datetime(tx["transaction_date"])
    tx["year"] = tx["date"].dt.year
    tx["month"] = tx["date"].dt.month

    tx_agg = tx.groupby(["department_id", "year", "month"])["amount"].sum().reset_index()

    merged = pd.merge(
        budgets,
        tx_agg,
        left_on=["department_id", "budget_year", "budget_month"],
        right_on=["department_id", "year", "month"],
        how="inner"
    )

    assert len(merged) == len(budgets), "Not all department-months matched between budgets and transactions!"

    # Maximum absolute discrepancy must be < 1e-4 INR
    discrepancy = np.abs(merged["actual_amount"] - merged["amount"])
    assert discrepancy.max() < 1e-4, f"Budget reconciliation failed! Max discrepancy: INR {discrepancy.max()}"


def test_budget_snapshots_monotonicity(v2_datasets):
    """Verify that Day 7 MTD <= Day 14 MTD <= Day 21 MTD <= Final Actual Amount."""
    snaps = v2_datasets["budget_snapshots.csv"]

    for (d_id, yr, mo), group in snaps.groupby(["department_id", "budget_year", "budget_month"]):
        group_sorted = group.sort_values("as_of_day")
        mtds = group_sorted["month_to_date_actual"].tolist()
        final_actual = group_sorted["final_actual_amount"].iloc[0]

        assert len(mtds) == 3, f"Expected 3 snapshots per department-month, got {len(mtds)}"
        assert mtds[0] <= mtds[1] <= mtds[2] <= final_actual + 1e-4, (
            f"Snapshot monotonicity violated for dept {d_id} {yr}-{mo}: {mtds} vs final {final_actual}"
        )


# ==============================================================================
# 6. VENDOR BILLING LIFECYCLE TESTS (USE CASE E)
# ==============================================================================

def test_vendor_billing_payment_lifecycle(v2_datasets):
    """Verify realistic payment date population and delay distribution in bills.csv."""
    bills = v2_datasets["bills.csv"]

    # In V1, payment_date was 100% null. In V2, ~90% must have payment dates.
    paid_count = bills["payment_date"].notna().sum()
    paid_rate = paid_count / len(bills)
    assert 0.85 <= paid_rate <= 0.95, f"Expected paid rate ~90%, got {paid_rate:.2%}"

    # Both on-time and late payments must exist
    late_count = bills["is_late_payment"].sum()
    assert late_count > 1000, f"Expected > 1,000 late payments, found {late_count}"

    # Status distribution
    statuses = set(bills["payment_status"].unique())
    assert {"PAID", "PENDING", "OVERDUE"}.issubset(statuses)


def test_invoices_duplicate_handling(v2_datasets):
    """Verify realistic duplicate invoice handling in invoices.csv."""
    inv = v2_datasets["invoices.csv"]
    dup_count = inv["is_duplicate"].sum()
    assert 150 <= dup_count <= 400, f"Expected 150–400 duplicate invoices, found {dup_count}"

    # Duplicate invoices must have rejection reason
    dups = inv[inv["is_duplicate"] == True]
    assert (dups["rejection_reason"] == "DUPLICATE_SUBMISSION").all()


# ==============================================================================
# 7. DETERMINISTIC REPRODUCIBILITY TEST
# ==============================================================================

def test_generator_deterministic_reproducibility(tmp_path):
    """Verify that identical seeds produce bit-for-bit identical datasets."""
    cfg1 = GeneratorConfig(seed=123, num_departments=5, num_employees=50, num_vendors=20, num_transactions=200, num_invoices=50, num_bills=50, output_dir=str(tmp_path / "run1"))
    cfg2 = GeneratorConfig(seed=123, num_departments=5, num_employees=50, num_vendors=20, num_transactions=200, num_invoices=50, num_bills=50, output_dir=str(tmp_path / "run2"))

    gen1 = EnterpriseDataGeneratorV2(cfg1)
    gen2 = EnterpriseDataGeneratorV2(cfg2)

    m1 = gen1.run_all()
    m2 = gen2.run_all()

    for fname in m1["checksums"]:
        assert m1["checksums"][fname] == m2["checksums"][fname], f"Checksum mismatch on {fname} for identical seed!"
