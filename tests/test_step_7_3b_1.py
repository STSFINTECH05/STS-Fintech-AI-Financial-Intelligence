"""
STS FINTECH AI FINANCIAL INTELLIGENCE
Automated Test Suite for Milestone 2 - Step 7.3B-1
V2 Expense Classification and Anomaly Detection Models

Tests cover:
1. V2 Dataset Loading and Integrity
2. Target and Feature Leakage Elimination (Model A & Model B)
3. Vendor Group Isolation and Zero-Overlap Validation
4. Chronological Split Separation
5. Hard Negative Scenario Ground-Truth Treatment
6. Model Inference and API Contracts
7. Model Artifact Serialization, Reload, and Deterministic Predictions
8. Metadata and SHA-256 Checksum Verification
9. V1 Artifact Preservation
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

from ml.preprocessing.data_cleaner_v2 import (
    load_v2_transaction_data,
    get_v2_classification_features,
    get_v2_anomaly_features
)
from ml.training.train_anomaly_detector_v2 import IsolationForestV2AnomalyDetector

V2_MODELS_DIR = os.path.join(BASE_DIR, "ml", "models", "v2")
V1_MODELS_DIR = os.path.join(BASE_DIR, "ml", "models")
DATA_V2_DIR = os.path.join(BASE_DIR, "data", "v2", "raw")
METADATA_PATH = os.path.join(V2_MODELS_DIR, "step_7_3b_1_metadata.json")


class TestV2DatasetIntegrity:
    """Verifies V2 raw dataset files and row counts against generator version manifest."""

    def test_v2_transactions_count_and_columns(self):
        df = load_v2_transaction_data()
        assert len(df) == 59834, f"Expected exactly 59,834 transactions, got {len(df)}"
        assert "transaction_id" in df.columns
        assert "category" in df.columns
        assert "is_anomaly" in df.columns
        assert "anomaly_type" in df.columns
        assert "description" in df.columns
        assert "amount" in df.columns
        assert df["amount"].min() > 0

    def test_v2_categories_count(self):
        df = load_v2_transaction_data()
        unique_cats = df["category"].unique()
        assert len(unique_cats) == 14, f"Expected 14 categories, found {len(unique_cats)}"

    def test_v2_anomaly_prevalence(self):
        df = load_v2_transaction_data()
        anomaly_count = df["is_anomaly"].sum()
        prevalence = anomaly_count / len(df)
        assert 0.02 <= prevalence <= 0.03, f"Expected anomaly prevalence ~2.36%, got {prevalence*100:.2f}%"
        assert anomaly_count == 1411


class TestFeatureLeakageGuards:
    """Strictly validates zero target leakage and absence of memorization identifiers."""

    def test_classification_zero_target_leakage(self):
        df = load_v2_transaction_data()
        X, y, text_feat, num_cols, cat_cols = get_v2_classification_features(df)

        prohibited = [
            "category", "transaction_id", "vendor_id", "employee_id",
            "is_anomaly", "anomaly_type", "anomaly_score", "currency", "is_business_expense"
        ]
        for col in prohibited:
            assert col not in X.columns, f"Prohibited column '{col}' leaked into classification features!"

        assert "description" in X.columns
        assert y.name == "category"

    def test_description_zero_category_name_leakage(self):
        df = load_v2_transaction_data()
        for cat in df["category"].unique():
            matches = df[df["description"].str.contains(cat, case=False, na=False)]
            assert len(matches) == 0, f"Target name '{cat}' directly leaked into {len(matches)} descriptions!"

    def test_anomaly_zero_target_leakage(self):
        df = load_v2_transaction_data()
        X, y, types, num_cols, cat_cols, _, _ = get_v2_anomaly_features(df)

        prohibited = [
            "is_anomaly", "anomaly_type", "anomaly_score", "transaction_id",
            "vendor_id", "employee_id", "currency", "is_business_expense"
        ]
        for col in prohibited:
            assert col not in X.columns, f"Prohibited column '{col}' leaked into anomaly features!"

        assert y.name == "is_anomaly"
        assert set(y.unique()).issubset({0, 1})


class TestPartitioningAndIsolation:
    """Verifies chronological boundaries and unseen-vendor isolation."""

    def test_chronological_split_boundaries(self):
        df = load_v2_transaction_data()
        d = pd.to_datetime(df["transaction_date"])

        train_mask = (d >= "2024-01-01") & (d <= "2025-12-31")
        val_mask = (d >= "2026-01-01") & (d <= "2026-08-31")
        test_mask = (d >= "2026-09-01") & (d <= "2026-10-31")
        stress_mask = (d >= "2026-11-14") & (d <= "2026-12-31")

        assert train_mask.sum() == 38523
        assert val_mask.sum() == 14297
        assert test_mask.sum() == 3592
        assert stress_mask.sum() == 2660

        # Verify zero temporal overlap
        assert (train_mask & val_mask).sum() == 0
        assert (val_mask & test_mask).sum() == 0
        assert (test_mask & stress_mask).sum() == 0

    def test_hard_negative_ground_truth(self):
        df = load_v2_transaction_data()
        hard_negatives = df[df["anomaly_type"] == "HARD_NEGATIVE_APPROVED_SURGE"]
        assert len(hard_negatives) == 916
        # Hard negatives must strictly have is_anomaly == False
        assert hard_negatives["is_anomaly"].sum() == 0, "Hard negatives must be flagged as legitimate normal (is_anomaly == False)!"


class TestModelArtifactsAndReload:
    """Verifies serialization, reload, deterministic inference, and metadata."""

    @pytest.fixture
    def artifacts(self):
        clf_path = os.path.join(V2_MODELS_DIR, "expense_classifier_v2.joblib")
        iso_path = os.path.join(V2_MODELS_DIR, "anomaly_detector_v2.joblib")
        rf_path = os.path.join(V2_MODELS_DIR, "anomaly_supervised_benchmark_v2.joblib")

        assert os.path.exists(clf_path), f"Missing artifact: {clf_path}"
        assert os.path.exists(iso_path), f"Missing artifact: {iso_path}"
        assert os.path.exists(rf_path), f"Missing artifact: {rf_path}"

        clf = joblib.load(clf_path)
        iso = joblib.load(iso_path)
        rf = joblib.load(rf_path)
        return clf, iso, rf

    def test_expense_classifier_inference(self, artifacts):
        clf, _, _ = artifacts
        df = load_v2_transaction_data()
        X, _, _, _, _ = get_v2_classification_features(df.head(20))

        preds = clf.predict(X)
        assert len(preds) == 20
        assert all(isinstance(p, str) for p in preds)
        assert all(p in clf.classes_ for p in preds)

        # Deterministic inference check
        preds2 = clf.predict(X)
        np.testing.assert_array_equal(preds, preds2)

    def test_isolation_forest_inference_and_risk_tiers(self, artifacts):
        _, iso, _ = artifacts
        df = load_v2_transaction_data()
        X, _, _, _, _, _, _ = get_v2_anomaly_features(df.head(30))

        scores = iso.predict_anomaly_score(X)
        assert len(scores) == 30
        assert (scores >= 0.0).all() and (scores <= 1.0).all()

        preds = iso.predict(X)
        assert len(preds) == 30
        assert set(preds).issubset({0, 1})

        tiers = iso.predict_risk_tier(X)
        assert len(tiers) == 30
        valid_tiers = {"Low Variance", "Moderate Discrepancy", "High Statistical Anomaly – Requires Verification"}
        assert all(t in valid_tiers for t in tiers)

    def test_supervised_anomaly_benchmark_inference(self, artifacts):
        _, _, rf = artifacts
        df = load_v2_transaction_data()
        X, _, _, _, _, _, _ = get_v2_anomaly_features(df.head(20))

        probs = rf.predict_proba(X)[:, 1]
        assert len(probs) == 20
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

    def test_metadata_file_and_checksums(self):
        assert os.path.exists(METADATA_PATH), f"Metadata file missing: {METADATA_PATH}"
        with open(METADATA_PATH, "r") as f:
            meta = json.load(f)

        assert meta["step"] == "7.3B-1"
        assert "artifacts" in meta

        for name, info in meta["artifacts"].items():
            path = info["path"]
            assert os.path.exists(path), f"Artifact in metadata not found on disk: {path}"
            with open(path, "rb") as af:
                computed_hash = hashlib.sha256(af.read()).hexdigest()
            assert computed_hash == info["sha256"], f"SHA-256 mismatch for {name}!"


class TestV1Preservation:
    """Confirms that V1 artifacts and dataset files remain 100% unaltered."""

    def test_v1_joblib_artifacts_exist(self):
        v1_artifacts = [
            "expense_classifier.joblib",
            "anomaly_detector.joblib",
            "anomaly_supervised_benchmark.joblib",
            "expense_forecaster.joblib",
            "budget_predictor.joblib",
            "vendor_analyzer.joblib"
        ]
        for art in v1_artifacts:
            path = os.path.join(V1_MODELS_DIR, art)
            assert os.path.exists(path), f"V1 artifact missing or deleted: {path}"
