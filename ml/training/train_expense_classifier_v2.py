"""
STS FINTECH AI FINANCIAL INTELLIGENCE
Model A: V2 Expense Classification Training Pipeline (Milestone 2 - Step 7.3B-1)

Predicts enterprise expense categories (14 classes) using transaction attributes,
merchant narratives, and organizational context.
Enforces strict zero-leakage standards:
- Strictly excludes 'category', 'transaction_id', 'vendor_id', 'employee_id',
  'is_anomaly', 'anomaly_type', and 'anomaly_score' from predictive input features.
- Text narrative processing via word & character-wb TF-IDF FeatureUnion.
- Evaluates LogisticRegression vs SGDClassifier vs LinearSVC.
- Chronological evaluation:
    * Train: 2024-01-01 to 2025-12-31 (38,523 transactions)
    * Validation: 2026-01-01 to 2026-08-31 (14,297 transactions) - Used strictly for model selection
    * Test (Historical Holdout): 2026-09-01 to 2026-10-31 (3,592 transactions)
    * Synthetic Future Stress Test: 2026-11-14 to 2026-12-31 (2,660 transactions) - Reported separately
- Unseen-Vendor Group Holdout: 20% of vendors (400 vendors) strictly held out.
- Department-Aware Holdout: 20% of departments (10 departments) held out.
- Persists production candidate to ml/models/v2/expense_classifier_v2.joblib.
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
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from ml.preprocessing.data_cleaner_v2 import load_v2_transaction_data, get_v2_classification_features
from ml.preprocessing.feature_pipeline_v2 import build_v2_classification_preprocessor

V2_MODELS_DIR = os.path.join(BASE_DIR, "ml", "models", "v2")
os.makedirs(V2_MODELS_DIR, exist_ok=True)


def train_expense_classifier_v2(
    random_state: int = 42
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Trains, validates, and comprehensively evaluates Model A (V2 Expense Classification).
    """
    print("=" * 70)
    print("STS FINTECH - MODEL A: V2 EXPENSE CLASSIFICATION TRAINING")
    print("=" * 70)

    # 1. Load data and extract leakage-safe features
    df = load_v2_transaction_data(enrich_vendors=True, enrich_departments=True, enrich_employees=True)
    X, y, text_feat, num_cols, cat_cols = get_v2_classification_features(df)

    print(f"Total Transactions: {len(df):,}")
    print(f"Feature count: {X.shape[1]}")
    print(f"Text feature: {text_feat}")
    print(f"Numeric features ({len(num_cols)}): {num_cols}")
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
    classes = sorted(y.unique())
    print(f"Target Categories ({len(classes)}): {classes}")

    # 2. Chronological Partitioning
    d = pd.to_datetime(df["transaction_date"])
    train_mask = (d >= "2024-01-01") & (d <= "2025-12-31")
    val_mask = (d >= "2026-01-01") & (d <= "2026-08-31")
    test_mask = (d >= "2026-09-01") & (d <= "2026-10-31")
    future_stress_mask = (d >= "2026-11-14") & (d <= "2026-12-31")

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    X_stress, y_stress = X[future_stress_mask], y[future_stress_mask]

    print("\n--- Chronological Split Distribution ---")
    print(f"  Training (2024-2025):              {len(X_train):,} ({len(X_train)/len(df)*100:.1f}%)")
    print(f"  Validation (2026 Jan-Aug):         {len(X_val):,} ({len(X_val)/len(df)*100:.1f}%)")
    print(f"  Historical Test (2026 Sep-Oct):    {len(X_test):,} ({len(X_test)/len(df)*100:.1f}%)")
    print(f"  Synthetic Future Stress (Nov-Dec): {len(X_stress):,} ({len(X_stress)/len(df)*100:.1f}%)")

    # Majority class baseline on validation
    majority_class = y_train.mode()[0]
    val_majority_baseline = float((y_val == majority_class).mean())
    print(f"  Majority Class Baseline ({majority_class}): {val_majority_baseline*100:.2f}%")

    # 3. Model Candidates Pipeline Setup
    preprocessor = build_v2_classification_preprocessor(text_feat, num_cols, cat_cols)

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
        ]),
        "LinearSVC": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LinearSVC(
                dual=False,
                class_weight="balanced",
                max_iter=2000,
                random_state=random_state
            ))
        ])
    }

    print("\n--- Candidate Evaluation & Selection on Validation Set ---")
    best_name = None
    best_pipeline = None
    best_val_macro_f1 = -1.0
    val_selection_results = {}

    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        y_val_pred = pipeline.predict(X_val)

        acc = float(accuracy_score(y_val, y_val_pred))
        macro_f1 = float(f1_score(y_val, y_val_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_val, y_val_pred, average="weighted", zero_division=0))

        val_selection_results[name] = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1
        }
        print(f"  {name:20s} | Val Accuracy: {acc*100:.2f}% | Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}")

        if macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = macro_f1
            best_name = name
            best_pipeline = pipeline

    print(f"\nOptimal Candidate Selected: {best_name} (Validation Macro F1: {best_val_macro_f1:.4f})")

    # 4. Final Evaluation on Untouched Historical Test Set (2026 Sep-Oct)
    y_test_pred = best_pipeline.predict(X_test)
    test_acc = float(accuracy_score(y_test, y_test_pred))
    test_macro_f1 = float(f1_score(y_test, y_test_pred, average="macro", zero_division=0))
    test_weighted_f1 = float(f1_score(y_test, y_test_pred, average="weighted", zero_division=0))
    test_cm = confusion_matrix(y_test, y_test_pred, labels=classes).tolist()
    test_cr_dict = classification_report(y_test, y_test_pred, target_names=classes, output_dict=True, zero_division=0)

    print("\n--- Final Evaluation: Historical Holdout Test (2026 Sep-Oct) ---")
    print(f"  Test Accuracy:     {test_acc*100:.2f}%")
    print(f"  Test Macro F1:     {test_macro_f1:.4f}")
    print(f"  Test Weighted F1:  {test_weighted_f1:.4f}")

    # 5. Synthetic Future Stress Test Evaluation (2026 Nov 14 - Dec 31)
    y_stress_pred = best_pipeline.predict(X_stress)
    stress_acc = float(accuracy_score(y_stress, y_stress_pred))
    stress_macro_f1 = float(f1_score(y_stress, y_stress_pred, average="macro", zero_division=0))
    stress_weighted_f1 = float(f1_score(y_stress, y_stress_pred, average="weighted", zero_division=0))

    print("\n--- Synthetic Future Stress Test (2026 Nov 14 - Dec 31) ---")
    print(f"  Stress Test Samples: {len(X_stress):,}")
    print(f"  Stress Accuracy:    {stress_acc*100:.2f}%")
    print(f"  Stress Macro F1:    {stress_macro_f1:.4f}")
    print(f"  Stress Weighted F1: {stress_weighted_f1:.4f}")

    # 6. Unseen-Vendor Group Evaluation
    print("\n--- Unseen-Vendor Generalization Evaluation ---")
    gss_vendor = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
    train_v_idx, test_v_idx = next(gss_vendor.split(X, y, groups=df["vendor_id"]))

    X_train_v, y_train_v = X.iloc[train_v_idx], y.iloc[train_v_idx]
    X_test_v, y_test_v = X.iloc[test_v_idx], y.iloc[test_v_idx]

    train_v_count = df["vendor_id"].iloc[train_v_idx].nunique()
    test_v_count = df["vendor_id"].iloc[test_v_idx].nunique()
    print(f"  Training Vendors: {train_v_count} ({len(X_train_v):,} rows)")
    print(f"  Held-Out Vendors: {test_v_count} ({len(X_test_v):,} rows) - Zero overlap")

    vendor_pipeline = Pipeline([
        ("preprocessor", build_v2_classification_preprocessor(text_feat, num_cols, cat_cols)),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state))
    ])
    vendor_pipeline.fit(X_train_v, y_train_v)
    y_test_v_pred = vendor_pipeline.predict(X_test_v)

    unseen_v_acc = float(accuracy_score(y_test_v, y_test_v_pred))
    unseen_v_macro_f1 = float(f1_score(y_test_v, y_test_v_pred, average="macro", zero_division=0))
    unseen_v_weighted_f1 = float(f1_score(y_test_v, y_test_v_pred, average="weighted", zero_division=0))
    print(f"  Unseen-Vendor Accuracy:    {unseen_v_acc*100:.2f}%")
    print(f"  Unseen-Vendor Macro F1:    {unseen_v_macro_f1:.4f}")
    print(f"  Unseen-Vendor Weighted F1: {unseen_v_weighted_f1:.4f}")

    # 7. Department-Aware Generalization Evaluation
    print("\n--- Department-Aware Generalization Evaluation ---")
    gss_dept = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
    train_d_idx, test_d_idx = next(gss_dept.split(X, y, groups=df["department_id"]))

    X_train_d, y_train_d = X.iloc[train_d_idx], y.iloc[train_d_idx]
    X_test_d, y_test_d = X.iloc[test_d_idx], y.iloc[test_d_idx]

    dept_pipeline = Pipeline([
        ("preprocessor", build_v2_classification_preprocessor(text_feat, num_cols, cat_cols)),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state))
    ])
    dept_pipeline.fit(X_train_d, y_train_d)
    y_test_d_pred = dept_pipeline.predict(X_test_d)

    dept_acc = float(accuracy_score(y_test_d, y_test_d_pred))
    dept_macro_f1 = float(f1_score(y_test_d, y_test_d_pred, average="macro", zero_division=0))
    dept_weighted_f1 = float(f1_score(y_test_d, y_test_d_pred, average="weighted", zero_division=0))
    print(f"  Department Holdout Accuracy: {dept_acc*100:.2f}%")
    print(f"  Department Holdout Macro F1: {dept_macro_f1:.4f}")

    # 8. Save production artifact to ml/models/v2/
    model_artifact_path = os.path.join(V2_MODELS_DIR, "expense_classifier_v2.joblib")
    joblib.dump(best_pipeline, model_artifact_path)

    with open(model_artifact_path, "rb") as f:
        artifact_hash = hashlib.sha256(f.read()).hexdigest()

    print(f"\nSaved V2 Expense Classifier to: {model_artifact_path}")
    print(f"Artifact SHA-256: {artifact_hash}")

    metrics = {
        "model_version": "2.0.0",
        "selected_model": best_name,
        "artifact_path": model_artifact_path,
        "sha256": artifact_hash,
        "training_timestamp": datetime.now().isoformat(),
        "classes": classes,
        "majority_class_baseline": val_majority_baseline,
        "features": {
            "text": text_feat,
            "numeric": num_cols,
            "categorical": cat_cols,
            "excluded_prohibited": [
                "category", "transaction_id", "vendor_id", "employee_id",
                "is_anomaly", "anomaly_type", "anomaly_score", "currency", "is_business_expense"
            ]
        },
        "validation_selection": val_selection_results,
        "historical_test_evaluation": {
            "sample_count": len(X_test),
            "accuracy": test_acc,
            "macro_f1": test_macro_f1,
            "weighted_f1": test_weighted_f1,
            "confusion_matrix": test_cm,
            "classification_report": test_cr_dict
        },
        "synthetic_future_stress_test": {
            "sample_count": len(X_stress),
            "accuracy": stress_acc,
            "macro_f1": stress_macro_f1,
            "weighted_f1": stress_weighted_f1
        },
        "unseen_vendor_evaluation": {
            "held_out_vendors_count": test_v_count,
            "test_sample_count": len(X_test_v),
            "accuracy": unseen_v_acc,
            "macro_f1": unseen_v_macro_f1,
            "weighted_f1": unseen_v_weighted_f1
        },
        "department_holdout_evaluation": {
            "accuracy": dept_acc,
            "macro_f1": dept_macro_f1,
            "weighted_f1": dept_weighted_f1
        }
    }

    return best_pipeline, metrics


if __name__ == "__main__":
    train_expense_classifier_v2()
