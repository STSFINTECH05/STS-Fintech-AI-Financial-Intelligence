"""
STS FINTECH AI FINANCIAL INTELLIGENCE
Feature Pipeline Transformers (Milestone 2 - Step 7.2B)

Constructs leakage-resistant ColumnTransformer preprocessors that fit strictly
on training data folds and transform validation/test sets consistently.
"""

from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder


def build_classification_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer for Model A (Expense Classification).
    - Standardizes continuous numeric variables.
    - One-hot encodes categorical variables with handle_unknown='ignore'
      to gracefully process previously unseen department or vendor categories.
    """
    transformers = []

    if numeric_features:
        transformers.append((
            "num",
            StandardScaler(),
            numeric_features
        ))

    if categorical_features:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_features
        ))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor


def build_anomaly_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer for Model B (Anomaly Detection).
    - Uses RobustScaler for numeric variables to prevent extreme spending spikes
      (anomalies) from distorting scaling centers and scale factors.
    - One-hot encodes categorical metadata.
    """
    transformers = []

    if numeric_features:
        transformers.append((
            "num",
            RobustScaler(),
            numeric_features
        ))

    if categorical_features:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_features
        ))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor
