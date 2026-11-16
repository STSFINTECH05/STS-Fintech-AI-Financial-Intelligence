"""
STS Fintech – Milestone 2 – Step 7.2D: Comprehensive Evaluation Pipeline
Evaluates Budget Predictor (Model D) and Vendor Analyzer (Model E).
Generates performance metrics, confusion matrices, data limitation manifests,
and persists verified SHA-256 checksums to ml/models/step_7_2d_metadata.json.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import hashlib
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support, confusion_matrix, classification_report
from ml.training.train_budget_predictor import aggregate_monthly_budgets, engineer_budget_features, audit_budget_transaction_reconciliation, FEATURE_COLS, classify_risk_tier
from ml.training.train_vendor_analyzer import audit_vendor_relationships, aggregate_vendor_metrics, compute_vendor_risk_score, SCORING_WEIGHTS


BUDGETS_PATH = "data/raw/budgets.csv"
TRANSACTIONS_PATH = "data/raw/transactions.csv"
BUDGET_MODEL_PATH = "ml/models/budget_predictor.joblib"
VENDOR_MODEL_PATH = "ml/models/vendor_analyzer.joblib"
METADATA_PATH = "ml/models/step_7_2d_metadata.json"


def calculate_sha256(filepath: str) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def evaluate_step_7_2d():
    print("=" * 80)
    print("STS FINTECH – MILESTONE 2 – STEP 7.2D COMPREHENSIVE EVALUATION")
    print("=" * 80)

    # ---------------------------------------------------------
    # PART 1: EVALUATE MODEL D (BUDGET PREDICTOR)
    # ---------------------------------------------------------
    print("\n>>> EVALUATING MODEL D: BUDGET PREDICTOR")
    budgets_df = pd.read_csv(BUDGETS_PATH)
    transactions_df = pd.read_csv(TRANSACTIONS_PATH)

    reconciliation = audit_budget_transaction_reconciliation(budgets_df, transactions_df)
    agg_df = aggregate_monthly_budgets(budgets_df)
    feat_df = engineer_budget_features(agg_df)

    train_mask = feat_df["period"] <= "2026-04"
    val_mask = (feat_df["period"] >= "2026-05") & (feat_df["period"] <= "2026-08")
    test_mask = feat_df["period"] >= "2026-09"

    X_train = feat_df.loc[train_mask, FEATURE_COLS].values.astype(np.float64)
    y_train = feat_df.loc[train_mask, "utilization_percentage"].values.astype(np.float64)
    X_val = feat_df.loc[val_mask, FEATURE_COLS].values.astype(np.float64)
    y_val = feat_df.loc[val_mask, "utilization_percentage"].values.astype(np.float64)
    X_test = feat_df.loc[test_mask, FEATURE_COLS].values.astype(np.float64)
    y_test = feat_df.loc[test_mask, "utilization_percentage"].values.astype(np.float64)

    # Load budget model
    budget_bundle = joblib.load(BUDGET_MODEL_PATH)
    budget_model = budget_bundle["model"]
    uses_scaler = budget_bundle.get("uses_scaler", False)
    scaler = budget_bundle.get("scaler")

    # Evaluate on Test Set
    if uses_scaler and scaler is not None:
        X_test_trans = scaler.transform(X_test)
        y_pred_test = budget_model.predict(X_test_trans)
    else:
        y_pred_test = budget_model.predict(X_test)

    y_pred_test = np.clip(y_pred_test, 0.0, 200.0)

    # Test Metrics
    budget_mae = float(mean_absolute_error(y_test, y_pred_test))
    budget_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    budget_wape = float((np.sum(np.abs(y_test - y_pred_test)) / np.sum(y_test)) * 100.0)

    # Baseline comparison (Dept Hist Mean)
    hist_idx = FEATURE_COLS.index("hist_mean_util")
    y_pred_base_test = X_test[:, hist_idx]
    base_mae = float(mean_absolute_error(y_test, y_pred_base_test))
    base_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_base_test)))
    base_wape = float((np.sum(np.abs(y_test - y_pred_base_test)) / np.sum(y_test)) * 100.0)

    # Overspending (>100%) evaluation
    actual_over = (y_test > 100.0).astype(int)
    pred_over = (y_pred_test > 100.0).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(actual_over, pred_over, average="binary", zero_division=0)
    cm_over = confusion_matrix(actual_over, pred_over).tolist()

    # Risk tiers
    actual_tiers = [classify_risk_tier(u) for u in y_test]
    pred_tiers = [classify_risk_tier(u) for u in y_pred_test]
    tier_report = classification_report(actual_tiers, pred_tiers, output_dict=True, zero_division=0)

    print(f"Model D Algorithm: {budget_bundle.get('model_name')}")
    print(f"Test MAE: {budget_mae:.2f}% (Baseline: {base_mae:.2f}%)")
    print(f"Test RMSE: {budget_rmse:.2f}% (Baseline: {base_rmse:.2f}%)")
    print(f"Test WAPE: {budget_wape:.2f}% (Baseline: {base_wape:.2f}%)")
    print(f"Overspending (>100%) Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print(f"Overspending Confusion Matrix: {cm_over}")

    # ---------------------------------------------------------
    # PART 2: EVALUATE MODEL E (VENDOR ANALYZER)
    # ---------------------------------------------------------
    print("\n>>> EVALUATING MODEL E: VENDOR ANALYZER")
    vendor_audit = audit_vendor_relationships()
    vendor_bundle = joblib.load(VENDOR_MODEL_PATH)
    profiles = vendor_bundle["profiles"]
    tier_dist = vendor_bundle["tier_distribution"]

    scores = [p["risk_score"] for p in profiles.values()]
    high_risk_vendors = [p for p in profiles.values() if p["risk_tier"] == "HIGH – REQUIRES VERIFICATION"]

    print(f"Vendor Screening Engine: {vendor_bundle.get('algorithm')}")
    print(f"Total Vendors Screened: {len(profiles):,}")
    print(f"Tier Breakdown: {tier_dist}")
    print(f"Risk Score Mean: {np.mean(scores):.2f}, Std: {np.std(scores):.2f}, Min: {np.min(scores):.2f}, Max: {np.max(scores):.2f}")
    print(f"High Risk Vendors Requiring Accounts Payable Verification: {len(high_risk_vendors):,}")

    # ---------------------------------------------------------
    # PART 3: ARTIFACT CHECKSUMS & METADATA PERSISTENCE
    # ---------------------------------------------------------
    budget_checksum = calculate_sha256(BUDGET_MODEL_PATH)
    vendor_checksum = calculate_sha256(VENDOR_MODEL_PATH)

    metadata = {
        "step": "Milestone 2 – Step 7.2D",
        "timestamp": datetime.utcnow().isoformat(),
        "model_d_budget_prediction": {
            "model_name": budget_bundle.get("model_name"),
            "artifact_path": BUDGET_MODEL_PATH,
            "sha256_checksum": budget_checksum,
            "dataset_source": BUDGETS_PATH,
            "original_allocation_lines": len(budgets_df),
            "aggregated_department_months": len(agg_df),
            "date_split_boundaries": {
                "training_period": ["2025-01", "2026-04"],
                "validation_period": ["2026-05", "2026-08"],
                "test_period": ["2026-09", "2026-12"]
            },
            "sample_counts": {
                "train_samples": int(len(X_train)),
                "val_samples": int(len(X_val)),
                "test_samples": int(len(X_test))
            },
            "features_used": FEATURE_COLS,
            "excluded_leakage_features": [
                "actual_amount",
                "remaining_amount",
                "utilization_percentage",
                "is_overspent"
            ],
            "test_performance": {
                "MAE": budget_mae,
                "RMSE": budget_rmse,
                "WAPE": budget_wape,
                "baseline_historical_mean_MAE": base_mae,
                "baseline_historical_mean_RMSE": base_rmse,
                "baseline_historical_mean_WAPE": base_wape,
                "overspending_classification": {
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1_score": float(f1),
                    "confusion_matrix": cm_over
                },
                "tier_classification": tier_report
            },
            "reconciliation_audit": reconciliation
        },
        "model_e_vendor_analysis": {
            "model_name": vendor_bundle.get("algorithm"),
            "artifact_path": VENDOR_MODEL_PATH,
            "sha256_checksum": vendor_checksum,
            "scoring_weights": SCORING_WEIGHTS,
            "scoring_thresholds": vendor_bundle.get("scoring_thresholds"),
            "dataset_sources": {
                "vendors": "data/raw/vendors.csv (2,000 records)",
                "invoices": "data/raw/invoices.csv (5,000 records)",
                "bills": "data/raw/bills.csv (5,000 records)",
                "transactions": "data/raw/transactions.csv (10,000 records)"
            },
            "data_quality_findings": {
                "foreign_key_integrity": vendor_audit["foreign_key_integrity"]["status"],
                "identifier_uniqueness": vendor_audit["identifier_uniqueness"],
                "bills_missing_payment_date": vendor_audit["bills_missing_payment_date"],
                "synthetic_risk_label_origin": vendor_audit["synthetic_risk_level_audit"]
            },
            "screening_results": {
                "total_vendors": len(profiles),
                "tier_distribution": tier_dist,
                "high_risk_count": len(high_risk_vendors),
                "risk_score_summary": {
                    "mean": round(float(np.mean(scores)), 2),
                    "std": round(float(np.std(scores)), 2),
                    "min": round(float(np.min(scores)), 2),
                    "max": round(float(np.max(scores)), 2)
                }
            },
            "compliance_policy": "Never classify a vendor as fraudulent based solely on statistical score. HIGH indicates accounts payable review required."
        }
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nPersisted Step 7.2D Metadata to {METADATA_PATH}")
    print(f"Model D SHA-256: {budget_checksum}")
    print(f"Model E SHA-256: {vendor_checksum}")
    print("=" * 80)
    return metadata


if __name__ == "__main__":
    evaluate_step_7_2d()
