"""
STS FINTECH AI FINANCIAL INTELLIGENCE
Model A: Expense Classification Training Pipeline (Milestone 2 - Step 7.2B)

Predicts expense categories (12 classes) using transaction attributes.
Enforces strict target leakage prevention by excluding 'description' and unique IDs.
Splits data into Train (70%), Validation (15%), and Test (15%) with stratification.
Evaluates LogisticRegression vs SGDClassifier, selects the optimal candidate on Validation,
and reports final evaluation on the untouched Test set.
Persists the complete Pipeline to ml/models/expense_classifier.joblib.
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
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from ml.preprocessing.data_cleaner import load_transaction_data, get_classification_features
from ml.preprocessing.feature_pipeline import build_classification_preprocessor


MODELS_DIR = os.path.join(BASE_DIR, "ml", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def train_expense_classifier(
    random_state: int = 42
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Trains and evaluates Model A (Expense Classification).
    """
    print("=" * 60)
    print("STS FINTECH - MODEL A: EXPENSE CLASSIFICATION TRAINING")
    print("=" * 60)

    # 1. Load data and extract leakage-safe features
    df = load_transaction_data(enrich_vendors=True)
    X, y, num_cols, cat_cols = get_classification_features(df)

    print(f"Total Transactions: {len(df):,}")
    print(f"Features ({len(X.columns)}): {list(X.columns)}")
    print(f"Numeric ({len(num_cols)}): {num_cols}")
    print(f"Categorical ({len(cat_cols)}): {cat_cols}")
    print(f"Target Categories ({y.nunique()}): {sorted(y.unique())}")

    # 2. Reproducible, leakage-resistant 70/15/15 stratified split
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
    print(f"  Training Samples:   {len(X_train):,} ({len(X_train)/len(df)*100:.1f}%)")
    print(f"  Validation Samples: {len(X_val):,} ({len(X_val)/len(df)*100:.1f}%)")
    print(f"  Test Samples:       {len(X_test):,} ({len(X_test)/len(df)*100:.1f}%)")

    # 3. Build preprocessing pipeline
    preprocessor = build_classification_preprocessor(num_cols, cat_cols)

    # 4. Model candidates
    candidates = {
        "LogisticRegression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=random_state
            ))
        ]),
        "SGDClassifier": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-4,
                class_weight="balanced",
                max_iter=1000,
                random_state=random_state
            ))
        ])
    }

    print("\n--- Model Selection on Validation Set ---")
    best_name = None
    best_pipeline = None
    best_val_f1 = -1.0
    val_results = {}

    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        y_val_pred = pipeline.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_macro_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
        val_weighted_f1 = f1_score(y_val, y_val_pred, average="weighted", zero_division=0)

        val_results[name] = {
            "accuracy": float(val_acc),
            "macro_f1": float(val_macro_f1),
            "weighted_f1": float(val_weighted_f1)
        }
        print(f"[{name}] Val Acc: {val_acc:.4f} | Val Macro F1: {val_macro_f1:.4f} | Val Weighted F1: {val_weighted_f1:.4f}")

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_name = name
            best_pipeline = pipeline

    print(f"\nOptimal Candidate Selected: {best_name} (Validation Macro F1: {best_val_f1:.4f})")

    # 5. Final Evaluation on Untouched Test Set
    print("\n--- Final Evaluation on Untouched Test Set ---")
    y_test_pred = best_pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_macro_f1 = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
    test_weighted_f1 = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)
    test_macro_prec = precision_score(y_test, y_test_pred, average="macro", zero_division=0)
    test_macro_rec = recall_score(y_test, y_test_pred, average="macro", zero_division=0)

    print(f"Test Accuracy:    {test_acc:.4f}")
    print(f"Test Macro F1:    {test_macro_f1:.4f}")
    print(f"Test Weighted F1: {test_weighted_f1:.4f}")
    print(f"Test Macro Prec:  {test_macro_prec:.4f}")
    print(f"Test Macro Rec:   {test_macro_rec:.4f}")

    labels = sorted(y.unique())
    clf_report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_test_pred, labels=labels)

    # 6. Save serialized pipeline
    model_path = os.path.join(MODELS_DIR, "expense_classifier.joblib")
    joblib.dump(best_pipeline, model_path)
    print(f"\nModel artifact serialized to: {model_path}")

    # Compute SHA-256 checksum
    with open(model_path, "rb") as f:
        sha256_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"SHA-256 Checksum: {sha256_hash}")

    metadata = {
        "model_name": "expense_classifier",
        "model_version": "1.0.0",
        "algorithm": best_name,
        "dataset_source": "data/raw/transactions.csv (enriched with vendors.csv)",
        "training_timestamp": datetime.now().isoformat(),
        "sha256_checksum": sha256_hash,
        "features": list(X.columns),
        "excluded_leakage_columns": [
            "description",
            "transaction_id",
            "category",
            "is_anomaly",
            "anomaly_score",
            "currency",
            "is_business_expense",
            "employee_id"
        ],
        "target_variable": "category",
        "target_classes": labels,
        "sample_counts": {
            "total": len(df),
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test)
        },
        "hyperparameters": best_pipeline.named_steps["classifier"].get_params(),
        "validation_comparison": val_results,
        "test_metrics": {
            "accuracy": float(test_acc),
            "macro_f1": float(test_macro_f1),
            "weighted_f1": float(test_weighted_f1),
            "macro_precision": float(test_macro_prec),
            "macro_recall": float(test_macro_rec),
            "per_class_metrics": {
                label: {
                    "precision": float(clf_report[label]["precision"]),
                    "recall": float(clf_report[label]["recall"]),
                    "f1-score": float(clf_report[label]["f1-score"]),
                    "support": int(clf_report[label]["support"])
                } for label in labels if label in clf_report
            },
            "confusion_matrix": cm.tolist()
        },
        "caveat": "Trained on synthetic enterprise data with leaked description excluded. Performance reflects synthetic generation properties and does not reflect real-world production text categorization."
    }

    return best_pipeline, metadata


if __name__ == "__main__":
    train_expense_classifier()
