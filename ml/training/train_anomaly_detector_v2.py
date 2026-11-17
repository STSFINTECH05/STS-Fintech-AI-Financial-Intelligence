"""
STS FINTECH AI FINANCIAL INTELLIGENCE
Model B: V2 Anomaly Detection Training Pipeline (Milestone 2 - Step 7.3B-1)

Implements two complementary financial anomaly detection approaches:
1. Unsupervised Baseline: IsolationForestV2 (fits on unlabeled normal operations; scores outliers)
2. Supervised Benchmark: RandomForestClassifier with balanced class weighting

Strict zero-leakage standards:
- Strictly excludes 'is_anomaly', 'anomaly_type', and 'anomaly_score' from predictive input features.
- 'anomaly_type' is retained solely as a ground-truth evaluation label for scenario-wise audit.
- Features include financial contextual ratios:
    * Amount to category median (fitted strictly on training set)
    * Amount to department median (fitted strictly on training set)
    * Amount to employee spending authorization limit (detects split threshold evasion)
    * Low vendor rating indicator & weekend timing signals
- Evaluates:
    * Chronological Holdout (Train: 2024-2025, Val: 2026 Jan-Aug, Test: 2026 Sep-Oct)
    * Threshold calibration performed strictly on Validation datafold
    * Scenario-wise recall across 4 anomaly patterns
    * Hard-negative false-positive rate on HARD_NEGATIVE_APPROVED_SURGE
    * Unseen-vendor generalization
    * Synthetic Future Stress Test (2026 Nov 14 - Dec 31)
- Persists artifacts to:
    * ml/models/v2/anomaly_detector_v2.joblib
    * ml/models/v2/anomaly_supervised_benchmark_v2.joblib
"""

import os
import sys
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from ml.preprocessing.data_cleaner_v2 import load_v2_transaction_data, get_v2_anomaly_features
from ml.preprocessing.feature_pipeline_v2 import build_v2_anomaly_preprocessor

V2_MODELS_DIR = os.path.join(BASE_DIR, "ml", "models", "v2")
os.makedirs(V2_MODELS_DIR, exist_ok=True)


class IsolationForestV2AnomalyDetector:
    """
    Production wrapper for V2 IsolationForest with preprocessing,
    min-max normalization, calibrated decision threshold, and 3-tier risk classification.
    """
    def __init__(
        self,
        preprocessor,
        n_estimators: int = 150,
        contamination: float = 0.025,
        random_state: int = 42
    ):
        self.preprocessor = preprocessor
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.min_score_ = None
        self.max_score_ = None
        self.threshold_ = 0.50

    def fit(self, X: pd.DataFrame):
        X_trans = self.preprocessor.fit_transform(X)
        self.model.fit(X_trans)
        raw_scores = -self.model.score_samples(X_trans)
        self.min_score_ = float(np.min(raw_scores))
        self.max_score_ = float(np.max(raw_scores))
        return self

    def predict_anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """Returns normalized anomaly score in range [0.0, 1.0]."""
        X_trans = self.preprocessor.transform(X)
        raw_scores = -self.model.score_samples(X_trans)
        denom = (self.max_score_ - self.min_score_) if (self.max_score_ != self.min_score_) else 1.0
        norm_scores = (raw_scores - self.min_score_) / denom
        return np.clip(norm_scores, 0.0, 1.0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Binary prediction (1 = anomaly, 0 = normal)."""
        scores = self.predict_anomaly_score(X)
        return (scores >= self.threshold_).astype(int)

    def predict_risk_tier(self, X: pd.DataFrame) -> np.ndarray:
        """Structured 3-tier advisory risk classification."""
        scores = self.predict_anomaly_score(X)
        labels = []
        for s in scores:
            if s < self.threshold_:
                labels.append("Low Variance")
            elif s < min(self.threshold_ + 0.20, 0.90):
                labels.append("Moderate Discrepancy")
            else:
                labels.append("High Statistical Anomaly – Requires Verification")
        return np.array(labels)


def train_anomaly_detectors_v2(
    random_state: int = 42
) -> Tuple[IsolationForestV2AnomalyDetector, Pipeline, Dict[str, Any]]:
    """
    Trains, calibrates, and evaluates both Unsupervised and Supervised V2 Anomaly Detectors.
    """
    print("=" * 70)
    print("STS FINTECH - MODEL B: V2 ANOMALY DETECTION TRAINING")
    print("=" * 70)

    # 1. Load data
    df = load_v2_transaction_data(enrich_vendors=True, enrich_departments=True, enrich_employees=True)
    d = pd.to_datetime(df["transaction_date"])

    # 2. Chronological masks
    train_mask = (d >= "2024-01-01") & (d <= "2025-12-31")
    val_mask = (d >= "2026-01-01") & (d <= "2026-08-31")
    test_mask = (d >= "2026-09-01") & (d <= "2026-10-31")
    future_stress_mask = (d >= "2026-11-14") & (d <= "2026-12-31")

    # Group medians calculated strictly on training fold to prevent distribution leakage
    cat_meds = df[train_mask].groupby("category")["amount"].median().to_dict()
    dept_meds = df[train_mask].groupby("department_id")["amount"].median().to_dict()

    X, y, anomaly_types, num_cols, cat_cols, _, _ = get_v2_anomaly_features(
        df, category_medians=cat_meds, department_medians=dept_meds
    )

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val, types_val = X[val_mask], y[val_mask], anomaly_types[val_mask]
    X_test, y_test, types_test = X[test_mask], y[test_mask], anomaly_types[test_mask]
    X_stress, y_stress, types_stress = X[future_stress_mask], y[future_stress_mask], anomaly_types[future_stress_mask]

    print(f"Total Transactions: {len(df):,}")
    print(f"Train samples:      {len(X_train):,} (Anomalies: {y_train.sum()} = {y_train.mean()*100:.2f}%)")
    print(f"Val samples:        {len(X_val):,} (Anomalies: {y_val.sum()} = {y_val.mean()*100:.2f}%)")
    print(f"Test samples:       {len(X_test):,} (Anomalies: {y_test.sum()} = {y_test.mean()*100:.2f}%)")
    print(f"Stress samples:     {len(X_stress):,} (Anomalies: {y_stress.sum()} = {y_stress.mean()*100:.2f}%)")

    # =========================================================================
    # MODEL B1: Unsupervised IsolationForest Baseline
    # =========================================================================
    print("\n--- Training Model B1: Unsupervised IsolationForest Baseline ---")
    iso_preprocessor = build_v2_anomaly_preprocessor(num_cols, cat_cols)
    iso_detector = IsolationForestV2AnomalyDetector(
        preprocessor=iso_preprocessor,
        n_estimators=150,
        contamination=0.025,
        random_state=random_state
    )
    iso_detector.fit(X_train)

    # Calibrate decision threshold on Validation set
    val_iso_scores = iso_detector.predict_anomaly_score(X_val)
    val_iso_pr_auc = float(average_precision_score(y_val, val_iso_scores))
    val_iso_roc_auc = float(roc_auc_score(y_val, val_iso_scores))

    best_iso_thresh = 0.50
    best_iso_f1 = 0.0
    for t in np.linspace(0.15, 0.85, 71):
        p = (val_iso_scores >= t).astype(int)
        f = f1_score(y_val, p, zero_division=0)
        if f > best_iso_f1:
            best_iso_f1 = float(f)
            best_iso_thresh = float(t)

    iso_detector.threshold_ = best_iso_thresh
    print(f"  Calibrated IsoForest Threshold on Val: {best_iso_thresh:.4f} (Val F1: {best_iso_f1:.4f})")
    print(f"  Val PR-AUC: {val_iso_pr_auc:.4f} | ROC-AUC: {val_iso_roc_auc:.4f}")

    # Evaluate IsoForest on Historical Test Set
    test_iso_scores = iso_detector.predict_anomaly_score(X_test)
    test_iso_preds = iso_detector.predict(X_test)

    iso_prec = float(precision_score(y_test, test_iso_preds, zero_division=0))
    iso_rec = float(recall_score(y_test, test_iso_preds, zero_division=0))
    iso_f1 = float(f1_score(y_test, test_iso_preds, zero_division=0))
    iso_pr_auc = float(average_precision_score(y_test, test_iso_scores))
    iso_roc_auc = float(roc_auc_score(y_test, test_iso_scores))
    iso_cm = confusion_matrix(y_test, test_iso_preds).tolist()

    print("\n[Model B1: IsolationForest Test Evaluation]")
    print(f"  Precision: {iso_prec:.4f} | Recall: {iso_rec:.4f} | F1: {iso_f1:.4f}")
    print(f"  PR-AUC:    {iso_pr_auc:.4f} | ROC-AUC: {iso_roc_auc:.4f}")
    print(f"  Confusion Matrix: {iso_cm}")

    # Scenario recall and hard negative evaluation for IsoForest
    iso_scenario_recalls = {}
    for sc in ["UNUSUAL_LARGE_AMOUNT", "SPLIT_APPROVAL_THRESHOLD", "OFF_HOURS_WEEKEND_OUTLIER", "UNRATED_OR_HIGH_RISK_VENDOR"]:
        sc_mask = (types_test == sc)
        if sc_mask.sum() > 0:
            sc_rec = float(test_iso_preds[sc_mask].mean())
            iso_scenario_recalls[sc] = {"count": int(sc_mask.sum()), "recall": sc_rec}
            print(f"    Scenario {sc} ({sc_mask.sum()} samples): Recall = {sc_rec*100:.2f}%")

    iso_hn_mask = (types_test == "HARD_NEGATIVE_APPROVED_SURGE")
    iso_hn_fpr = float(test_iso_preds[iso_hn_mask].mean()) if iso_hn_mask.sum() > 0 else 0.0
    print(f"    Hard-Negative Surge FPR: {iso_hn_fpr*100:.2f}% ({test_iso_preds[iso_hn_mask].sum()}/{iso_hn_mask.sum()} false alerts)")

    # =========================================================================
    # MODEL B2: Supervised RandomForest Benchmark
    # =========================================================================
    print("\n--- Training Model B2: Supervised RandomForest Benchmark ---")
    rf_preprocessor = build_v2_anomaly_preprocessor(num_cols, cat_cols)
    rf_model = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1
    )
    rf_pipeline = Pipeline([
        ("preprocessor", rf_preprocessor),
        ("classifier", rf_model)
    ])
    rf_pipeline.fit(X_train, y_train)

    val_rf_probs = rf_pipeline.predict_proba(X_val)[:, 1]
    val_rf_pr_auc = float(average_precision_score(y_val, val_rf_probs))
    val_rf_roc_auc = float(roc_auc_score(y_val, val_rf_probs))

    # Calibrate decision threshold on Validation set
    best_rf_thresh = 0.50
    best_rf_f1 = 0.0
    for t in np.arange(0.15, 0.85, 0.02):
        p = (val_rf_probs >= t).astype(int)
        f = f1_score(y_val, p, zero_division=0)
        if f > best_rf_f1:
            best_rf_f1 = float(f)
            best_rf_thresh = float(t)

    print(f"  Calibrated RF Threshold on Val: {best_rf_thresh:.4f} (Val F1: {best_rf_f1:.4f})")
    print(f"  Val PR-AUC: {val_rf_pr_auc:.4f} | ROC-AUC: {val_rf_roc_auc:.4f}")

    # Evaluate RF on Historical Test Set
    test_rf_probs = rf_pipeline.predict_proba(X_test)[:, 1]
    test_rf_preds = (test_rf_probs >= best_rf_thresh).astype(int)

    rf_prec = float(precision_score(y_test, test_rf_preds, zero_division=0))
    rf_rec = float(recall_score(y_test, test_rf_preds, zero_division=0))
    rf_f1 = float(f1_score(y_test, test_rf_preds, zero_division=0))
    rf_pr_auc = float(average_precision_score(y_test, test_rf_probs))
    rf_roc_auc = float(roc_auc_score(y_test, test_rf_probs))
    rf_cm = confusion_matrix(y_test, test_rf_preds).tolist()

    print("\n[Model B2: Supervised RandomForest Test Evaluation]")
    print(f"  Precision: {rf_prec:.4f} | Recall: {rf_rec:.4f} | F1: {rf_f1:.4f}")
    print(f"  PR-AUC:    {rf_pr_auc:.4f} | ROC-AUC: {rf_roc_auc:.4f}")
    print(f"  Confusion Matrix: {rf_cm}")

    rf_scenario_recalls = {}
    for sc in ["UNUSUAL_LARGE_AMOUNT", "SPLIT_APPROVAL_THRESHOLD", "OFF_HOURS_WEEKEND_OUTLIER", "UNRATED_OR_HIGH_RISK_VENDOR"]:
        sc_mask = (types_test == sc)
        if sc_mask.sum() > 0:
            sc_rec = float(test_rf_preds[sc_mask].mean())
            rf_scenario_recalls[sc] = {"count": int(sc_mask.sum()), "recall": sc_rec}
            print(f"    Scenario {sc} ({sc_mask.sum()} samples): Recall = {sc_rec*100:.2f}%")

    rf_hn_mask = (types_test == "HARD_NEGATIVE_APPROVED_SURGE")
    rf_hn_fpr = float(test_rf_preds[rf_hn_mask].mean()) if rf_hn_mask.sum() > 0 else 0.0
    print(f"    Hard-Negative Surge FPR: {rf_hn_fpr*100:.2f}% ({test_rf_preds[rf_hn_mask].sum()}/{rf_hn_mask.sum()} false alerts)")

    # =========================================================================
    # Synthetic Future Stress Test Evaluation
    # =========================================================================
    stress_rf_probs = rf_pipeline.predict_proba(X_stress)[:, 1]
    stress_rf_preds = (stress_rf_probs >= best_rf_thresh).astype(int)
    stress_rf_prec = float(precision_score(y_stress, stress_rf_preds, zero_division=0))
    stress_rf_rec = float(recall_score(y_stress, stress_rf_preds, zero_division=0))
    stress_rf_f1 = float(f1_score(y_stress, stress_rf_preds, zero_division=0))
    stress_rf_pr_auc = float(average_precision_score(y_stress, stress_rf_probs))
    stress_rf_roc_auc = float(roc_auc_score(y_stress, stress_rf_probs))

    # =========================================================================
    # Unseen-Vendor Generalization Evaluation
    # =========================================================================
    print("\n--- Unseen-Vendor Anomaly Detection Evaluation ---")
    gss_v = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
    train_v_idx, test_v_idx = next(gss_v.split(X, y, groups=df["vendor_id"]))

    X_train_v, y_train_v = X.iloc[train_v_idx], y.iloc[train_v_idx]
    X_test_v, y_test_v = X.iloc[test_v_idx], y.iloc[test_v_idx]

    unseen_rf_pipeline = Pipeline([
        ("preprocessor", build_v2_anomaly_preprocessor(num_cols, cat_cols)),
        ("classifier", RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample", random_state=random_state, n_jobs=-1))
    ])
    unseen_rf_pipeline.fit(X_train_v, y_train_v)
    unseen_probs = unseen_rf_pipeline.predict_proba(X_test_v)[:, 1]
    unseen_preds = (unseen_probs >= best_rf_thresh).astype(int)

    unseen_prec = float(precision_score(y_test_v, unseen_preds, zero_division=0))
    unseen_rec = float(recall_score(y_test_v, unseen_preds, zero_division=0))
    unseen_f1 = float(f1_score(y_test_v, unseen_preds, zero_division=0))
    unseen_pr_auc = float(average_precision_score(y_test_v, unseen_probs))
    unseen_roc_auc = float(roc_auc_score(y_test_v, unseen_probs))

    print(f"  Unseen-Vendor RF Precision: {unseen_prec:.4f} | Recall: {unseen_rec:.4f} | F1: {unseen_f1:.4f}")
    print(f"  Unseen-Vendor RF PR-AUC:    {unseen_pr_auc:.4f} | ROC-AUC: {unseen_roc_auc:.4f}")

    # =========================================================================
    # 3. Save Model Artifacts
    # =========================================================================
    iso_path = os.path.join(V2_MODELS_DIR, "anomaly_detector_v2.joblib")
    rf_path = os.path.join(V2_MODELS_DIR, "anomaly_supervised_benchmark_v2.joblib")

    joblib.dump(iso_detector, iso_path)
    joblib.dump(rf_pipeline, rf_path)

    with open(iso_path, "rb") as f:
        iso_hash = hashlib.sha256(f.read()).hexdigest()
    with open(rf_path, "rb") as f:
        rf_hash = hashlib.sha256(f.read()).hexdigest()

    print(f"\nSaved V2 IsolationForest Baseline to: {iso_path} (SHA-256: {iso_hash})")
    print(f"Saved V2 Supervised RandomForest to:   {rf_path} (SHA-256: {rf_hash})")

    metrics = {
        "model_version": "2.0.0",
        "training_timestamp": datetime.now().isoformat(),
        "unsupervised_baseline": {
            "model_type": "IsolationForestV2AnomalyDetector",
            "artifact_path": iso_path,
            "sha256": iso_hash,
            "calibrated_threshold": best_iso_thresh,
            "validation_metrics": {
                "f1": best_iso_f1,
                "pr_auc": val_iso_pr_auc,
                "roc_auc": val_iso_roc_auc
            },
            "historical_test_metrics": {
                "sample_count": len(X_test),
                "anomalies_count": int(y_test.sum()),
                "precision": iso_prec,
                "recall": iso_rec,
                "f1": iso_f1,
                "pr_auc": iso_pr_auc,
                "roc_auc": iso_roc_auc,
                "confusion_matrix": iso_cm,
                "scenario_recalls": iso_scenario_recalls,
                "hard_negative_surge_fpr": iso_hn_fpr
            }
        },
        "supervised_benchmark": {
            "model_type": "RandomForestClassifier",
            "artifact_path": rf_path,
            "sha256": rf_hash,
            "calibrated_threshold": best_rf_thresh,
            "validation_metrics": {
                "f1": best_rf_f1,
                "pr_auc": val_rf_pr_auc,
                "roc_auc": val_rf_roc_auc
            },
            "historical_test_metrics": {
                "sample_count": len(X_test),
                "anomalies_count": int(y_test.sum()),
                "precision": rf_prec,
                "recall": rf_rec,
                "f1": rf_f1,
                "pr_auc": rf_pr_auc,
                "roc_auc": rf_roc_auc,
                "confusion_matrix": rf_cm,
                "scenario_recalls": rf_scenario_recalls,
                "hard_negative_surge_fpr": rf_hn_fpr
            },
            "synthetic_future_stress_test": {
                "sample_count": len(X_stress),
                "anomalies_count": int(y_stress.sum()),
                "precision": stress_rf_prec,
                "recall": stress_rf_rec,
                "f1": stress_rf_f1,
                "pr_auc": stress_rf_pr_auc,
                "roc_auc": stress_rf_roc_auc
            },
            "unseen_vendor_evaluation": {
                "test_sample_count": len(X_test_v),
                "precision": unseen_prec,
                "recall": unseen_rec,
                "f1": unseen_f1,
                "pr_auc": unseen_pr_auc,
                "roc_auc": unseen_roc_auc
            }
        },
        "feature_specification": {
            "numeric_features": num_cols,
            "categorical_features": cat_cols,
            "excluded_prohibited": [
                "is_anomaly", "anomaly_type", "anomaly_score", "transaction_id",
                "vendor_id", "employee_id", "currency", "is_business_expense"
            ]
        }
    }

    return iso_detector, rf_pipeline, metrics


if __name__ == "__main__":
    train_anomaly_detectors_v2()
