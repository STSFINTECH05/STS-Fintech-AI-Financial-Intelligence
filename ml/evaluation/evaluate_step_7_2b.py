"""
STS FINTECH AI FINANCIAL INTELLIGENCE
Comprehensive Evaluation & Metadata Serialization (Milestone 2 - Step 7.2B)

Coordinates Model A (Expense Classification) and Model B (Anomaly Detection),
generates unified evaluation reports, validates model artifacts and checksums,
and saves ml/models/model_metadata.json.
"""

import os
import sys
import json
import hashlib
import platform
import sklearn
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from ml.training.train_expense_classifier import train_expense_classifier
from ml.training.train_anomaly_detector import train_anomaly_detectors


MODELS_DIR = os.path.join(BASE_DIR, "ml", "models")


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_full_step_7_2b_evaluation():
    print("=" * 70)
    print("STS FINTECH - MILESTONE 2 - STEP 7.2B EVALUATION RUNNER")
    print("=" * 70)

    # 1. Train & Evaluate Expense Classifier
    classifier_pipeline, clf_meta = train_expense_classifier(random_state=42)

    # 2. Train & Evaluate Anomaly Detectors
    iso_model, rf_model, anomaly_meta = train_anomaly_detectors(random_state=42)

    # 3. Verify on-disk artifacts and calculate fresh SHA-256 checksums
    clf_path = os.path.join(MODELS_DIR, "expense_classifier.joblib")
    iso_path = os.path.join(MODELS_DIR, "anomaly_detector.joblib")
    rf_path = os.path.join(MODELS_DIR, "anomaly_supervised_benchmark.joblib")

    for path in [clf_path, iso_path, rf_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected model artifact missing: {path}")

    clf_checksum = compute_sha256(clf_path)
    iso_checksum = compute_sha256(iso_path)
    rf_checksum = compute_sha256(rf_path)

    clf_meta["sha256_checksum"] = clf_checksum
    anomaly_meta["unsupervised_isolation_forest"]["sha256_checksum"] = iso_checksum
    anomaly_meta["supervised_benchmark"]["sha256_checksum"] = rf_checksum

    # 4. Assemble comprehensive metadata
    full_metadata = {
        "step": "Milestone 2 - Step 7.2B",
        "description": "Expense Classification & Anomaly Detection Baseline Models",
        "timestamp": clf_meta["training_timestamp"],
        "environment": {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "joblib_version": joblib.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "os": platform.platform()
        },
        "artifacts": {
            "expense_classifier": {
                "file": "expense_classifier.joblib",
                "sha256": clf_checksum,
                "size_bytes": os.path.getsize(clf_path)
            },
            "anomaly_detector": {
                "file": "anomaly_detector.joblib",
                "sha256": iso_checksum,
                "size_bytes": os.path.getsize(iso_path)
            },
            "anomaly_supervised_benchmark": {
                "file": "anomaly_supervised_benchmark.joblib",
                "sha256": rf_checksum,
                "size_bytes": os.path.getsize(rf_path)
            }
        },
        "model_a_expense_classification": clf_meta,
        "model_b_anomaly_detection": anomaly_meta
    }

    # 5. Persist model_metadata.json
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(full_metadata, f, indent=2)

    print(f"\n✅ All artifacts and metadata successfully written to: {MODELS_DIR}")
    print(f"Metadata file: {metadata_path}")

    return full_metadata


if __name__ == "__main__":
    run_full_step_7_2b_evaluation()
