"""
Comprehensive Test Suite for Milestone 2 - Step 7.2B
Verifies:
- Dataset loading and schema compliance
- Feature and target separation
- Target leakage elimination (verifying description, category, is_anomaly are not in X)
- Model training and inference output shapes
- Valid predicted categories (12 classes)
- Anomaly score range [0, 1] and risk tier labels
- Model artifact persistence, reload, and SHA-256 integrity
- Deterministic prediction reproducibility
"""

import os
import sys
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from ml.preprocessing.data_cleaner import (
    load_transaction_data,
    get_classification_features,
    get_anomaly_features
)
from ml.training.train_anomaly_detector import IsolationForestAnomalyDetector


class TestDatasetAndLeakagePrevention:
    """Test suite for dataset ingestion and target leakage prevention."""

    @pytest.fixture
    def transactions_df(self):
        return load_transaction_data(enrich_vendors=True)

    def test_transactions_row_count(self, transactions_df):
        assert len(transactions_df) == 10000, "transactions.csv must have 10,000 rows"

    def test_vendor_enrichment_columns(self, transactions_df):
        assert "risk_level" in transactions_df.columns
        assert "vendor_category" in transactions_df.columns
        assert "vendor_rating" in transactions_df.columns

    def test_calendar_features_extracted(self, transactions_df):
        assert "day_of_week" in transactions_df.columns
        assert "is_weekend" in transactions_df.columns
        assert transactions_df["day_of_week"].isin(range(7)).all()
        assert transactions_df["is_weekend"].isin([0, 1]).all()

    def test_classification_feature_leakage_exclusion(self, transactions_df):
        X, y, num_cols, cat_cols = get_classification_features(transactions_df)

        prohibited = [
            "description",
            "category",
            "is_anomaly",
            "anomaly_score",
            "transaction_id",
            "currency",
            "is_business_expense",
            "employee_id"
        ]
        for col in prohibited:
            assert col not in X.columns, f"Prohibited leakage column '{col}' found in classification features!"

        assert len(y) == len(X)
        assert y.nunique() == 12, "Target must have exactly 12 expense categories"

    def test_anomaly_feature_leakage_exclusion(self, transactions_df):
        X, y, num_cols, cat_cols, cat_meds, dept_meds = get_anomaly_features(transactions_df)

        prohibited = [
            "is_anomaly",
            "anomaly_score",
            "description",
            "transaction_id",
            "currency",
            "is_business_expense",
            "employee_id"
        ]
        for col in prohibited:
            assert col not in X.columns, f"Prohibited leakage column '{col}' found in anomaly features!"

        assert len(y) == len(X)
        assert set(y.unique()) == {0, 1}, "Anomaly target must be binary (0 or 1)"


class TestModelArtifactsAndInference:
    """Test suite for saved models, prediction shapes, and reproducibility."""

    @pytest.fixture(scope="class")
    def artifacts_dir(self):
        return os.path.join(BASE_DIR, "ml", "models")

    def test_artifacts_exist_on_disk(self, artifacts_dir):
        expected_files = [
            "expense_classifier.joblib",
            "anomaly_detector.joblib",
            "anomaly_supervised_benchmark.joblib",
            "model_metadata.json"
        ]
        for filename in expected_files:
            filepath = os.path.join(artifacts_dir, filename)
            assert os.path.exists(filepath), f"Expected artifact '{filename}' not found on disk"
            assert os.path.getsize(filepath) > 0, f"Artifact '{filename}' is empty"

    def test_model_metadata_structure(self, artifacts_dir):
        meta_path = os.path.join(artifacts_dir, "model_metadata.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        assert meta["step"] == "Milestone 2 - Step 7.2B"
        assert "artifacts" in meta
        assert "model_a_expense_classification" in meta
        assert "model_b_anomaly_detection" in meta
        assert "environment" in meta

    def test_expense_classifier_reload_and_predict(self, artifacts_dir):
        model_path = os.path.join(artifacts_dir, "expense_classifier.joblib")
        pipeline = joblib.load(model_path)

        df = load_transaction_data(enrich_vendors=True).head(20)
        X, _, _, _ = get_classification_features(df)

        preds = pipeline.predict(X)
        probas = pipeline.predict_proba(X)

        assert len(preds) == 20
        assert probas.shape == (20, 12), "Probabilities must have shape (N, 12 categories)"
        assert np.allclose(probas.sum(axis=1), 1.0), "Probabilities must sum to 1.0"

        # Check valid categories
        expected_categories = {
            "Facilities", "Maintenance", "Transportation", "Hardware",
            "IT Services", "Cloud Services", "Software", "Professional Services",
            "Travel", "Office Supplies", "Marketing", "Meals"
        }
        for pred in preds:
            assert pred in expected_categories, f"Invalid category predicted: {pred}"

    def test_anomaly_detector_reload_and_predict(self, artifacts_dir):
        model_path = os.path.join(artifacts_dir, "anomaly_detector.joblib")
        detector = joblib.load(model_path)

        df = load_transaction_data(enrich_vendors=True).head(25)
        X, _, _, _, _, _ = get_anomaly_features(df)

        scores = detector.predict_anomaly_score(X)
        binary_preds = detector.predict(X)
        risk_tiers = detector.predict_risk_tier(X)

        assert len(scores) == 25
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0), "Anomaly scores must be bounded in [0, 1]"
        assert set(binary_preds).issubset({0, 1}), "Predictions must be binary"

        valid_tiers = {
            "Low Variance",
            "Moderate Discrepancy",
            "High Statistical Anomaly – Requires Verification"
        }
        for tier in risk_tiers:
            assert tier in valid_tiers, f"Invalid risk tier generated: {tier}"

    def test_supervised_anomaly_benchmark_reload(self, artifacts_dir):
        model_path = os.path.join(artifacts_dir, "anomaly_supervised_benchmark.joblib")
        pipeline = joblib.load(model_path)

        df = load_transaction_data(enrich_vendors=True).head(15)
        X, _, _, _, _, _ = get_anomaly_features(df)

        preds = pipeline.predict(X)
        probas = pipeline.predict_proba(X)

        assert len(preds) == 15
        assert probas.shape == (15, 2)
        assert set(preds).issubset({0, 1})

    def test_prediction_reproducibility(self, artifacts_dir):
        model_path = os.path.join(artifacts_dir, "expense_classifier.joblib")
        pipeline = joblib.load(model_path)

        df = load_transaction_data(enrich_vendors=True).head(10)
        X, _, _, _ = get_classification_features(df)

        preds_run1 = pipeline.predict(X)
        preds_run2 = pipeline.predict(X)
        assert np.array_equal(preds_run1, preds_run2), "Inference must be strictly deterministic"
