"""
STS FINTECH AI FINANCIAL INTELLIGENCE
V2 Feature Pipeline Transformers (Milestone 2 - Step 7.3B-1)

Constructs leakage-resistant scikit-learn ColumnTransformer and FeatureUnion pipelines:
- V2 Expense Classification: Word & Character n-gram TF-IDF for descriptions,
  StandardScaler for numerical metrics, and OneHotEncoder for operational categories.
- V2 Anomaly Detection: RobustScaler for non-Gaussian/skewed financial distributions,
  and OneHotEncoder for enterprise metadata.
"""

from typing import List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder


def build_v2_classification_preprocessor(
    text_feature: str = "description",
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    word_max_features: int = 5000,
    char_max_features: int = 5000
) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer for Model A (V2 Expense Classification).
    - Extracts word n-grams (1-2) and character-wb n-grams (3-5) via FeatureUnion.
    - Standardizes continuous numeric variables.
    - One-hot encodes categorical metadata with handle_unknown='ignore'.
    """
    if numeric_features is None:
        numeric_features = []
    if categorical_features is None:
        categorical_features = []

    text_pipeline = FeatureUnion([
        (
            "word_tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=word_max_features,
                sublinear_tf=True
            )
        ),
        (
            "char_tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=char_max_features,
                sublinear_tf=True
            )
        )
    ])

    transformers = []

    # Text transformation
    if text_feature:
        # text column needs to be passed as 1D array or Series to TfidfVectorizer
        transformers.append((
            "text",
            text_pipeline,
            text_feature
        ))

    # Numeric transformation
    if numeric_features:
        transformers.append((
            "num",
            StandardScaler(),
            numeric_features
        ))

    # Categorical transformation
    if categorical_features:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            categorical_features
        ))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor


def build_v2_anomaly_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer for Model B (V2 Anomaly Detection).
    - Uses RobustScaler for numeric variables to resist distortion from extreme anomaly spikes.
    - One-hot encodes categorical variables (sparse_output=False for tree-based models).
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
