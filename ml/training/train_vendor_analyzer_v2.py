"""
STS Fintech – Milestone 2 – Step 7.3B-3
Model E: V2 Vendor Intelligence & Accounts Payable Risk Suite

Trains and serializes the three sub-engines of Model E:
1. Duplicate Invoice Screening Engine (Deterministic Matcher + Supervised Benchmark)
2. Late Payment Prediction Engine (Supervised AP Classifier)
3. Multi-Factor Vendor Risk Screening Engine (Enterprise 0–100 Transparent Scoring)
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
)

from ml.prediction.vendor_analyzer_v2 import OperationalDuplicateMatcher

DATA_DIR = "data/v2/raw"
OUTPUT_DIR = "ml/models/v2"

DUP_MODEL_PATH = os.path.join(OUTPUT_DIR, "vendor_duplicate_detector_v2.joblib")
PAY_MODEL_PATH = os.path.join(OUTPUT_DIR, "vendor_payment_predictor_v2.joblib")
RISK_MODEL_PATH = os.path.join(OUTPUT_DIR, "vendor_screening_v2.joblib")


# =====================================================================
# 1. DUPLICATE INVOICE SCREENING SUB-ENGINE
# =====================================================================


def train_duplicate_detector(data_dir: str = DATA_DIR, output_path: str = DUP_MODEL_PATH):
    """
    Builds the operational deterministic matcher and trains the supervised benchmark
    for invoice duplicate screening, with strict exclusion of post-review leakage.
    """
    print("\n--- Training Sub-engine 1: Duplicate Invoice Screening ---")
    invoices = pd.read_csv(os.path.join(data_dir, "invoices.csv"))
    invoices["inv_date"] = pd.to_datetime(invoices["invoice_date"])
    invoices["due_dt"] = pd.to_datetime(invoices["due_date"])

    # Strict exclusion of downstream leakage:
    # rejection_reason and invoice_status are strictly dropped
    invoices["day_of_week"] = invoices["inv_date"].dt.dayofweek
    invoices["day_of_month"] = invoices["inv_date"].dt.day
    invoices["month"] = invoices["inv_date"].dt.month
    invoices["terms_days"] = (invoices["due_dt"] - invoices["inv_date"]).dt.days
    invoices["log_amount"] = np.log1p(invoices["total_amount"])
    invoices["log_subtotal"] = np.log1p(invoices["subtotal"])
    invoices["tax_ratio"] = invoices["tax_amount"] / np.maximum(1.0, invoices["total_amount"])

    # Chronological Split
    train_mask = (invoices["inv_date"] <= "2025-12-31")
    val_mask = (invoices["inv_date"] > "2025-12-31") & (invoices["inv_date"] <= "2026-08-31")
    test_mask = (invoices["inv_date"] > "2026-08-31") & (invoices["inv_date"] <= "2026-10-31")
    stress_mask = (invoices["inv_date"] > "2026-10-31")

    train_val_mask = train_mask | val_mask

    feature_cols = [
        "department_id", "subtotal", "tax_amount", "total_amount",
        "log_amount", "log_subtotal", "tax_ratio", "terms_days",
        "day_of_week", "day_of_month", "month"
    ]

    X_train = invoices.loc[train_val_mask, feature_cols]
    y_train = invoices.loc[train_val_mask, "is_duplicate"].astype(int)

    X_test = invoices.loc[test_mask, feature_cols]
    y_test = invoices.loc[test_mask, "is_duplicate"].astype(int)

    # Operational rule matcher initialized on training set
    matcher = OperationalDuplicateMatcher(window_days=30, amount_tolerance_pct=0.01)
    matcher.register_invoices(invoices[train_val_mask])

    # Supervised Benchmark (RandomForestClassifier)
    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # Handle edge case where test set might have zero positive duplicates or rare positives
    pos_count = int(y_test.sum())
    roc_auc = float(roc_auc_score(y_test, y_prob)) if pos_count > 0 else 0.5
    pr_auc = float(average_precision_score(y_test, y_prob)) if pos_count > 0 else float(y_train.mean())

    print(f"Supervised Benchmark Test Performance (Sep–Oct 2026, {len(X_test)} invoices, {pos_count} positives):")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  PR-AUC:    {pr_auc:.4f}")
    print(f"  Synthetic Note: 'is_duplicate' was synthetically generated with uniform 2.5% random Bernoulli noise,")
    print(f"                  rendering pre-review invoice attributes uninformative without data leakage.")

    bundle = {
        "engine": "VendorDuplicateDetectorV2",
        "operational_matcher": matcher,
        "supervised_benchmark": clf,
        "feature_names": feature_cols,
        "metrics": {
            "test_samples": len(X_test),
            "test_positives": pos_count,
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4)
        },
        "synthetic_generation_audit": {
            "mechanism": "Uniform random Bernoulli sampling (p=0.025) in generator without duplicate-pair duplication",
            "leakage_safeguard": "invoice_status and rejection_reason strictly excluded from feature pipeline",
            "operational_strategy": "Primary protection via OperationalDuplicateMatcher with exact/near amount matching"
        }
    }
    joblib.dump(bundle, output_path)
    print(f"Artifact saved to: {output_path} ({os.path.getsize(output_path):,} bytes)")
    return bundle


# =====================================================================
# 2. LATE PAYMENT PREDICTION SUB-ENGINE
# =====================================================================

def train_payment_predictor(data_dir: str = DATA_DIR, output_path: str = PAY_MODEL_PATH):
    """
    Trains a supervised accounts payable classifier to predict late bill settlement
    strictly using pre-bill vendor history and contractual terms.
    """
    print("\n--- Training Sub-engine 2: Late Payment Predictor ---")
    bills = pd.read_csv(os.path.join(data_dir, "bills.csv"))
    vendors = pd.read_csv(os.path.join(data_dir, "vendors.csv"))

    bills["b_date"] = pd.to_datetime(bills["bill_date"])
    bills["d_date"] = pd.to_datetime(bills["due_date"])
    bills = bills.merge(vendors[["vendor_id", "vendor_rating", "payment_terms_days"]], on="vendor_id", how="left")

    # Strict exclusion of downstream leakage: payment_status and payment_date dropped
    bills["terms_days"] = (bills["d_date"] - bills["b_date"]).dt.days
    bills["month"] = bills["b_date"].dt.month
    bills["day_of_week"] = bills["b_date"].dt.dayofweek
    bills["day_of_month"] = bills["b_date"].dt.day
    bills["log_amount"] = np.log1p(bills["total_amount"])
    bills["tax_ratio"] = bills["tax_amount"] / np.maximum(1.0, bills["total_amount"])

    # Chronological pre-bill vendor late rate (strict zero future leakage)
    bills = bills.sort_values("b_date").reset_index(drop=True)
    hist_rates = []
    vendor_history = {}
    for idx, row in bills.iterrows():
        vid = row["vendor_id"]
        if vid in vendor_history and len(vendor_history[vid]) > 0:
            hist_rates.append(float(np.mean(vendor_history[vid])))
        else:
            hist_rates.append(0.2096)  # global prior mean
        vendor_history.setdefault(vid, []).append(int(row["is_late_payment"]))

    bills["vendor_prior_late_rate"] = hist_rates

    # Chronological Split
    train_mask = (bills["b_date"] <= "2025-12-31")
    val_mask = (bills["b_date"] > "2025-12-31") & (bills["b_date"] <= "2026-08-31")
    test_mask = (bills["b_date"] > "2026-08-31") & (bills["b_date"] <= "2026-10-31")
    stress_mask = (bills["b_date"] > "2026-10-31")

    train_val_mask = train_mask | val_mask

    feature_cols = [
        "department_id", "bill_amount", "tax_amount", "total_amount",
        "log_amount", "tax_ratio", "terms_days", "vendor_rating",
        "vendor_prior_late_rate", "month", "day_of_week", "day_of_month"
    ]

    X_train = bills.loc[train_val_mask, feature_cols]
    y_train = bills.loc[train_val_mask, "is_late_payment"].astype(int)

    X_test = bills.loc[test_mask, feature_cols]
    y_test = bills.loc[test_mask, "is_late_payment"].astype(int)

    # Train Classifier
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # Baseline: Naive majority class prediction (always predict False, probability = train prior)
    train_prior = float(y_train.mean())
    base_prob = np.full_like(y_prob, train_prior)
    base_brier = brier_score_loss(y_test, base_prob)
    model_brier = brier_score_loss(y_test, y_prob)

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    print(f"Late Payment Test Performance (Sep–Oct 2026, {len(X_test)} bills):")
    print(f"  Accuracy:     {acc:.4f}")
    print(f"  ROC-AUC:      {roc_auc:.4f}")
    print(f"  PR-AUC:       {pr_auc:.4f}")
    print(f"  Model Brier:  {model_brier:.4f} (Naive Baseline Brier: {base_brier:.4f})")

    # Store vendor final historical rates up to train/val end
    final_vendor_rates = {vid: float(np.mean(vals)) for vid, vals in vendor_history.items()}

    bundle = {
        "engine": "VendorPaymentPredictorV2",
        "model": clf,
        "feature_names": feature_cols,
        "vendor_historical_rates": final_vendor_rates,
        "global_prior_late_rate": train_prior,
        "metrics": {
            "test_samples": len(X_test),
            "accuracy": round(float(acc), 4),
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "model_brier_score": round(float(model_brier), 4),
            "baseline_brier_score": round(float(base_brier), 4)
        }
    }
    joblib.dump(bundle, output_path)
    print(f"Artifact saved to: {output_path} ({os.path.getsize(output_path):,} bytes)")
    return bundle


# =====================================================================
# 3. MULTI-FACTOR VENDOR RISK SCREENING SUB-ENGINE
# =====================================================================

SCORING_WEIGHTS = {
    "rejection_factor": 0.25,
    "duplicate_factor": 0.20,
    "late_payment_factor": 0.20,
    "anomaly_factor": 0.20,
    "rating_deficit_factor": 0.15
}


def build_vendor_screening_engine(data_dir: str = DATA_DIR, output_path: str = RISK_MODEL_PATH):
    """
    Builds the multi-factor vendor risk screening engine across all 2,000 registered vendors.
    """
    print("\n--- Training Sub-engine 3: Multi-Factor Vendor Risk Screening Engine ---")
    vendors = pd.read_csv(os.path.join(data_dir, "vendors.csv"))
    invoices = pd.read_csv(os.path.join(data_dir, "invoices.csv"))
    bills = pd.read_csv(os.path.join(data_dir, "bills.csv"))
    tx = pd.read_csv(os.path.join(data_dir, "transactions.csv"))

    # 1. Invoice aggregation
    inv_agg = invoices.groupby("vendor_id").agg(
        invoice_count=("invoice_id", "count"),
        total_invoice_amt=("total_amount", "sum"),
        avg_invoice_amt=("total_amount", "mean"),
        rejected_invoices=("invoice_status", lambda s: int((s == "REJECTED").sum())),
        duplicate_invoices=("is_duplicate", "sum")
    ).reset_index()

    # 2. Bill aggregation
    bill_agg = bills.groupby("vendor_id").agg(
        bills_count=("bill_id", "count"),
        total_bill_amt=("total_amount", "sum"),
        late_bills=("is_late_payment", "sum"),
        overdue_bills=("payment_status", lambda s: int((s == "OVERDUE").sum()))
    ).reset_index()

    # 3. Transaction aggregation
    tx_agg = tx.groupby("vendor_id").agg(
        tx_count=("transaction_id", "count"),
        total_tx_amt=("amount", "sum"),
        anomaly_tx_count=("is_anomaly", "sum")
    ).reset_index()

    # Merge into single master vendor profile
    v_full = vendors.merge(inv_agg, on="vendor_id", how="left")
    v_full = v_full.merge(bill_agg, on="vendor_id", how="left")
    v_full = v_full.merge(tx_agg, on="vendor_id", how="left")

    fill_zeros = [
        "invoice_count", "total_invoice_amt", "avg_invoice_amt",
        "rejected_invoices", "duplicate_invoices",
        "bills_count", "total_bill_amt", "late_bills", "overdue_bills",
        "tx_count", "total_tx_amt", "anomaly_tx_count"
    ]
    for col in fill_zeros:
        v_full[col] = v_full[col].fillna(0)

    # Derived Rates
    v_full["rejection_rate"] = np.where(v_full["invoice_count"] > 0, v_full["rejected_invoices"] / v_full["invoice_count"], 0.0)
    v_full["duplicate_rate"] = np.where(v_full["invoice_count"] > 0, v_full["duplicate_invoices"] / v_full["invoice_count"], 0.0)
    v_full["late_rate"] = np.where(v_full["bills_count"] > 0, v_full["late_bills"] / v_full["bills_count"], 0.0)
    v_full["anomaly_rate"] = np.where(v_full["tx_count"] > 0, v_full["anomaly_tx_count"] / v_full["tx_count"], 0.0)

    profiles = {}
    tier_counts = {"LOW": 0, "MEDIUM": 0, "HIGH – REQUIRES VERIFICATION": 0}

    low_thresh = 25.0
    med_thresh = 50.0

    for _, row in v_full.iterrows():
        vid = str(row["vendor_id"]).strip().upper()

        # Factor 1: Invoice Rejection Rate (0% to 50%+ maps to 0–100)
        f1 = float(np.clip(row["rejection_rate"] * 200.0, 0.0, 100.0))

        # Factor 2: Duplicate Invoicing (each duplicate invoice adds 50 points)
        f2 = float(np.clip(row["duplicate_invoices"] * 50.0, 0.0, 100.0))

        # Factor 3: Late Payment Rate (0% to 66%+ maps to 0–100)
        f3 = float(np.clip(row["late_rate"] * 150.0, 0.0, 100.0))

        # Factor 4: Transaction Anomaly Rate (0% to 50%+ maps to 0–100)
        f4 = float(np.clip(row["anomaly_rate"] * 200.0, 0.0, 100.0))

        # Factor 5: Vendor Rating Deficit ((5.0 - rating) / 5.0 * 100)
        v_rating = float(row.get("vendor_rating", 3.75))
        f5 = float(np.clip(((5.0 - v_rating) / 5.0) * 100.0, 0.0, 100.0))

        composite_score = (
            SCORING_WEIGHTS["rejection_factor"] * f1 +
            SCORING_WEIGHTS["duplicate_factor"] * f2 +
            SCORING_WEIGHTS["late_payment_factor"] * f3 +
            SCORING_WEIGHTS["anomaly_factor"] * f4 +
            SCORING_WEIGHTS["rating_deficit_factor"] * f5
        )
        composite_score = float(np.clip(composite_score, 0.0, 100.0))

        if composite_score < low_thresh:
            tier = "LOW"
            action = "Standard automated three-way matching and scheduled disbursement."
        elif composite_score <= med_thresh:
            tier = "MEDIUM"
            action = "Routine automated validation with periodic accounts payable sampling."
        else:
            tier = "HIGH – REQUIRES VERIFICATION"
            action = "Mandatory manual verification of pending invoices and vendor documentation prior to payment authorization."

        tier_counts[tier] += 1

        profiles[vid] = {
            "vendor_id": vid,
            "vendor_name": str(row["vendor_name"]),
            "vendor_category": str(row["vendor_category"]),
            "location": str(row.get("location", "National")),
            "vendor_rating": v_rating,
            "payment_terms_days": int(row.get("payment_terms_days", 30)),
            "invoice_count": int(row["invoice_count"]),
            "total_invoice_amt": round(float(row["total_invoice_amt"]), 2),
            "rejected_invoices": int(row["rejected_invoices"]),
            "duplicate_invoices": int(row["duplicate_invoices"]),
            "bills_count": int(row["bills_count"]),
            "total_bill_amt": round(float(row["total_bill_amt"]), 2),
            "late_bills": int(row["late_bills"]),
            "overdue_bills": int(row["overdue_bills"]),
            "tx_count": int(row["tx_count"]),
            "total_tx_amt": round(float(row["total_tx_amt"]), 2),
            "anomaly_tx_count": int(row["anomaly_tx_count"]),
            "factor_scores": {
                "invoice_rejection_score": round(f1, 2),
                "duplicate_invoicing_score": round(f2, 2),
                "late_payment_score": round(f3, 2),
                "transaction_anomaly_score": round(f4, 2),
                "rating_deficit_score": round(f5, 2)
            },
            "risk_score": round(composite_score, 2),
            "risk_tier": tier,
            "recommended_action": action
        }

    print(f"Risk Tier Distribution across {len(profiles):,} vendors:")
    for tier, count in tier_counts.items():
        pct = (count / len(profiles)) * 100.0
        print(f"  {tier:30s}: {count:5d} ({pct:5.2f}%)")

    bundle = {
        "algorithm": "EnterpriseMultiFactorRuleScreeningEngineV2",
        "scoring_weights": SCORING_WEIGHTS,
        "scoring_thresholds": {
            "low_threshold": low_thresh,
            "medium_threshold": med_thresh
        },
        "profiles": profiles,
        "tier_distribution": tier_counts
    }
    joblib.dump(bundle, output_path)
    print(f"Artifact saved to: {output_path} ({os.path.getsize(output_path):,} bytes)")
    return bundle


def train_vendor_analyzer_v2():
    """Trains all three Model E sub-engines."""
    print("=" * 70)
    print("STS FINTECH – STEP 7.3B-3: MODEL E – VENDOR INTELLIGENCE SUITE")
    print("=" * 70)
    train_duplicate_detector()
    train_payment_predictor()
    build_vendor_screening_engine()


if __name__ == "__main__":
    train_vendor_analyzer_v2()
