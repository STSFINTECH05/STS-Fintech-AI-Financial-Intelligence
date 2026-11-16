"""
STS FINTECH AI FINANCIAL INTELLIGENCE
Model B: Anomaly Detection Training Pipeline (Milestone 2 - Step 7.2B)

Implements two complementary financial anomaly detection approaches:
1. Unsupervised Baseline: IsolationForest (fits on unlabeled features; scores outliers)
2. Supervised Benchmark: RandomForestClassifier with balanced class weights

Strictly excludes 'is_anomaly' and 'anomaly_score' from predictive input features.
Evaluates both models on an untouched test set using Precision, Recall, F1 (anomaly class),
ROC-AUC, PR-AUC, and Confusion Matrix.
Maps output into 3 structured risk tiers:
- Low Variance
- Moderate Discrepancy
- High Statistical Anomaly – Requires Verification
Persists artifacts to ml/models/anomaly_detector.joblib and anomaly_supervised_benchmark.joblib.
"""

import os
import sys
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.pipeline import Pipeline
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

from ml.preprocessing.data_cleaner import load_transaction_data, get_anomaly_features
from ml.preprocessing.feature_pipeline import build_anomaly_preprocessor


MODELS_DIR = os.path.join(BASE_DIR, "ml", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


class IsolationForestAnomalyDetector:
    """
    Wrapper for IsolationForest with internal preprocessing, score normalization,
    and 3-tier risk classification.
    """
    def __init__(self, preprocessor, n_estimators=150, contamination=0.087, random_state=42):
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

    def fit(self, X):
        X_trans = self.preprocessor.fit_transform(X)
        self.model.fit(X_trans)
        # In scikit-learn, score_samples returns opposite of anomaly score (lower is more anomalous)
        raw_scores = -self.model.score_samples(X_trans)
        self.min_score_ = float(np.min(raw_scores))
        self.max_score_ = float(np.max(raw_scores))
        return self

    def predict_anomaly_score(self, X) -> np.ndarray:
        """Returns normalized anomaly score between 0.0 and 1.0."""
        X_trans = self.preprocessor.transform(X)
        raw_scores = -self.model.score_samples(X_trans)
        denom = (self.max_score_ - self.min_score_) if (self.max_score_ != self.min_score_) else 1.0
        norm_scores = (raw_scores - self.min_score_) / denom
        return np.clip(norm_scores, 0.0, 1.0)

    def predict(self, X) -> np.ndarray:
        """Binary prediction (1 = anomaly, 0 = normal)."""
        scores = self.predict_anomaly_score(X)
        return (scores >= self.threshold_).astype(int)

    def predict_risk_tier(self, X) -> np.ndarray:
        """Returns structured 3-tier operational risk classification."""
        scores = self.predict_anomaly_score(X)
        labels = []
        for s in scores:
            if s < 0.50:
                labels.append("Low Variance")
            elif s < 0.70:
                labels.append("Moderate Discrepancy")
            else:
                labels.append("High Statistical Anomaly – Requires Verification")
        return np.array(labels)


def train_anomaly_detectors(
    random_state: int = 42
) -> Tuple[IsolationForestAnomalyDetector, Pipeline, Dict[str, Any]]:
    """
    Trains and evaluates both Unsupervised and Supervised Anomaly Detectors.
    """
    print("=" * 60)
    print("STS FINTECH - MODEL B: ANOMALY DETECTION TRAINING")
    print("=" * 60)

    # 1. Load data and extract features
    df = load_transaction_data(enrich_vendors=True)
    X, y, num_cols, cat_cols, cat_meds, dept_meds = get_anomaly_features(df)

    anomaly_count = y.sum()
    anomaly_rate = y.mean() * 100
    print(f"Total Transactions: {len(df):,}")
    print(f"Ground Truth Anomalies: {anomaly_count:,} ({anomaly_rate:.2f}%)")
    print(f"Normal Transactions:   {len(df) - anomaly_count:,} ({100 - anomaly_rate:.2f}%)")
    print(f"Numeric Features ({len(num_cols)}): {num_cols}")
    print(f"Categorical Features ({len(cat_cols)}): {cat_cols}")

    # 2. Stratified 70/15/15 split on is_anomaly
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=random_state,
        stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp
    )

    print(f"\nSplit Distribution:")
    print(f"  Training:   {len(X_train):,} (Anomalies: {y_train.sum()})")
    print(f"  Validation: {len(X_val):,} (Anomalies: {y_val.sum()})")
    print(f"  Test:       {len(X_test):,} (Anomalies: {y_test.sum()})")

    # ----------------------------------------------------
    # APPROACH A: UNSUPERVISED ISOLATION FOREST
    # ----------------------------------------------------
    print("\n--- Training Approach A: Unsupervised IsolationForest ---")
    unsup_preprocessor = build_anomaly_preprocessor(num_cols, cat_cols)
    iso_detector = IsolationForestAnomalyDetector(
        preprocessor=unsup_preprocessor,
        n_estimators=150,
        contamination=0.087,
        random_state=random_state
    )
    # Fit strictly without labels
    iso_detector.fit(X_train)

    # Tune threshold on validation set to maximize F1 for anomaly class
    val_scores = iso_detector.predict_anomaly_score(X_val)
    best_thresh = 0.50
    best_thresh_f1 = -1.0
    for thresh in np.arange(0.35, 0.75, 0.02):
        pred = (val_scores >= thresh).astype(int)
        f1 = f1_score(y_val, pred, zero_division=0)
        if f1 > best_thresh_f1:
            best_thresh_f1 = f1
            best_thresh = thresh

    iso_detector.threshold_ = float(best_thresh)
    print(f"Calibrated Decision Threshold: {best_thresh:.2f} (Validation Anomaly F1: {best_thresh_f1:.4f})")

    # Evaluate on Untouched Test Set
    test_scores_iso = iso_detector.predict_anomaly_score(X_test)
    test_pred_iso = (test_scores_iso >= best_thresh).astype(int)

    iso_prec = precision_score(y_test, test_pred_iso, zero_division=0)
    iso_rec = recall_score(y_test, test_pred_iso, zero_division=0)
    iso_f1 = f1_score(y_test, test_pred_iso, zero_division=0)
    iso_roc_auc = roc_auc_score(y_test, test_scores_iso)
    iso_pr_auc = average_precision_score(y_test, test_scores_iso)
    cm_iso = confusion_matrix(y_test, test_pred_iso)

    print("\n[IsolationForest Test Results]")
    print(f"  Precision: {iso_prec:.4f}")
    print(f"  Recall:    {iso_rec:.4f}")
    print(f"  F1-Score:  {iso_f1:.4f}")
    print(f"  ROC-AUC:   {iso_roc_auc:.4f}")
    print(f"  PR-AUC:    {iso_pr_auc:.4f}")
    print(f"  Confusion Matrix (TN, FP, FN, TP):\n{cm_iso}")

    # ----------------------------------------------------
    # APPROACH B: SUPERVISED RANDOM FOREST BENCHMARK
    # ----------------------------------------------------
    print("\n--- Training Approach B: Supervised RandomForest Benchmark ---")
    sup_preprocessor = build_anomaly_preprocessor(num_cols, cat_cols)
    sup_pipeline = Pipeline([
        ("preprocessor", sup_preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ))
    ])

    sup_pipeline.fit(X_train, y_train)

    test_proba_rf = sup_pipeline.predict_proba(X_test)[:, 1]
    test_pred_rf = sup_pipeline.predict(X_test)

    rf_prec = precision_score(y_test, test_pred_rf, zero_division=0)
    rf_rec = recall_score(y_test, test_pred_rf, zero_division=0)
    rf_f1 = f1_score(y_test, test_pred_rf, zero_division=0)
    rf_roc_auc = roc_auc_score(y_test, test_proba_rf)
    rf_pr_auc = average_precision_score(y_test, test_proba_rf)
    cm_rf = confusion_matrix(y_test, test_pred_rf)

    print("\n[RandomForest Benchmark Test Results]")
    print(f"  Precision: {rf_prec:.4f}")
    print(f"  Recall:    {rf_rec:.4f}")
    print(f"  F1-Score:  {rf_f1:.4f}")
    print(f"  ROC-AUC:   {rf_roc_auc:.4f}")
    print(f"  PR-AUC:    {rf_pr_auc:.4f}")
    print(f"  Confusion Matrix (TN, FP, FN, TP):\n{cm_rf}")

    # ----------------------------------------------------
    # PERSISTENCE & SHA-256 HASHING
    # ----------------------------------------------------
    iso_path = os.path.join(MODELS_DIR, "anomaly_detector.joblib")
    rf_path = os.path.join(MODELS_DIR, "anomaly_supervised_benchmark.joblib")

    joblib.dump(iso_detector, iso_path)
    joblib.dump(sup_pipeline, rf_path)

    with open(iso_path, "rb") as f:
        iso_hash = hashlib.sha256(f.read()).hexdigest()
    with open(rf_path, "rb") as f:
        rf_hash = hashlib.sha256(f.read()).hexdigest()

    print(f"\nSaved IsolationForest artifact to: {iso_path}")
    print(f"  SHA-256: {iso_hash}")
    print(f"Saved Supervised RF artifact to:   {rf_path}")
    print(f"  SHA-256: {rf_hash}")

    metadata = {
        "model_name": "anomaly_detector",
        "model_version": "1.0.0",
        "training_timestamp": datetime.now().isoformat(),
        "unsupervised_isolation_forest": {
            "artifact": "anomaly_detector.joblib",
            "sha256_checksum": iso_hash,
            "algorithm": "IsolationForest",
            "hyperparameters": {
                "n_estimators": iso_detector.n_estimators,
                "contamination": iso_detector.contamination,
                "calibrated_threshold": iso_detector.threshold_,
                "min_score": iso_detector.min_score_,
                "max_score": iso_detector.max_score_
            },
            "test_metrics": {
                "precision": float(iso_prec),
                "recall": float(iso_rec),
                "f1_score": float(iso_f1),
                "roc_auc": float(iso_roc_auc),
                "pr_auc": float(iso_pr_auc),
                "confusion_matrix": cm_iso.tolist(),
                "false_positives": int(cm_iso[0][1]),
                "false_negatives": int(cm_iso[1][0]),
                "true_positives": int(cm_iso[1][1]),
                "true_negatives": int(cm_iso[0][0])
            }
        },
        "supervised_benchmark": {
            "artifact": "anomaly_supervised_benchmark.joblib",
            "sha256_checksum": rf_hash,
            "algorithm": "RandomForestClassifier(balanced)",
            "test_metrics": {
                "precision": float(rf_prec),
                "recall": float(rf_rec),
                "f1_score": float(rf_f1),
                "roc_auc": float(rf_roc_auc),
                "pr_auc": float(rf_pr_auc),
                "confusion_matrix": cm_rf.tolist(),
                "false_positives": int(cm_rf[0][1]),
                "false_negatives": int(cm_rf[1][0]),
                "true_positives": int(cm_rf[1][1]),
                "true_negatives": int(cm_rf[0][0])
            }
        },
        "risk_tiers": [
            {"tier": "Low Variance", "score_range": "< 0.50", "action": "Auto-cleared"},
            {"tier": "Moderate Discrepancy", "score_range": "0.50 - 0.69", "action": "Flagged for standard periodic review"},
            {"tier": "High Statistical Anomaly – Requires Verification", "score_range": ">= 0.70", "action": "Requires mandatory human audit verification"}
        ],
        "features": list(X.columns),
        "excluded_leakage_columns": [
            "is_anomaly",
            "anomaly_score",
            "transaction_id",
            "description",
            "currency",
            "is_business_expense",
            "employee_id"
        ],
        "sample_counts": {
            "total": len(df),
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
            "test_anomalies": int(y_test.sum())
        }
    }

    return iso_detector, sup_pipeline, metadata


if __name__ == "__main__":
    train_anomaly_detectors()
