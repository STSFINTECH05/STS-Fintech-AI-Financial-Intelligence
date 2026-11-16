"""
STS FINTECH AI FINANCIAL INTELLIGENCE
Data Cleaner and Feature Preparation (Milestone 2 - Step 7.2B)

Provides robust ingestion, joining, and feature extraction for:
- Model A: Expense Classification
- Model B: Anomaly Detection

Strictly enforces target leakage elimination by barring:
- 'description' (contains direct target string f"{category} expense")
- 'transaction_id' (unique ID leading to memorization)
- 'is_anomaly' and 'anomaly_score' (from predictive feature sets)
- 'category' (from Model A features)
"""

import os
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")


def load_transaction_data(
    transactions_csv: str = "transactions.csv",
    vendors_csv: Optional[str] = "vendors.csv",
    data_dir: str = DATA_DIR,
    enrich_vendors: bool = True
) -> pd.DataFrame:
    """
    Loads raw transaction data and optionally enriches it with vendor metadata.
    Extracts core calendar features from transaction_date.
    """
    tx_path = os.path.join(data_dir, transactions_csv) if not os.path.isabs(transactions_csv) else transactions_csv
    if not os.path.exists(tx_path):
        raise FileNotFoundError(f"Transactions dataset not found at: {tx_path}")

    df = pd.read_csv(tx_path)

    # Optional vendor metadata enrichment (preserves enterprise relationship graph)
    if enrich_vendors and vendors_csv:
        vn_path = os.path.join(data_dir, vendors_csv) if not os.path.isabs(vendors_csv) else vendors_csv
        if os.path.exists(vn_path):
            df_vn = pd.read_csv(vn_path)
            vn_cols = ["vendor_id", "risk_level", "vendor_category", "vendor_rating"]
            available_cols = [c for c in vn_cols if c in df_vn.columns]
            df = df.merge(df_vn[available_cols], on="vendor_id", how="left")
            # Fill missing vendor attributes if any unlinked vendor occurs
            if "risk_level" in df.columns:
                df["risk_level"] = df["risk_level"].fillna("MEDIUM")
            if "vendor_rating" in df.columns:
                df["vendor_rating"] = df["vendor_rating"].fillna(df["vendor_rating"].median())
            if "vendor_category" in df.columns:
                df["vendor_category"] = df["vendor_category"].fillna("General")

    # Calendar feature engineering
    if "transaction_date" in df.columns:
        dt = pd.to_datetime(df["transaction_date"], errors="coerce")
        df["day_of_week"] = dt.dt.weekday.fillna(0).astype(int)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["month"] = dt.dt.month.fillna(1).astype(int)
        df["day"] = dt.dt.day.fillna(1).astype(int)

    # Numerical transformations
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        df["log_amount"] = np.log1p(np.maximum(df["amount"], 0.0))

    return df


def get_classification_features(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Extracts features and target for Model A: Expense Classification.

    STRICT LEAKAGE GUARD:
    - Excludes 'description' (which contains category name directly)
    - Excludes 'transaction_id'
    - Excludes 'category' from X
    - Excludes 'is_anomaly' and 'anomaly_score'
    - Excludes constant columns ('currency', 'is_business_expense') and high-cardinality 'employee_id'

    Returns:
        X: Feature DataFrame
        y: Target Series (category)
        numeric_features: List of numeric column names in X
        categorical_features: List of categorical column names in X
    """
    if "category" not in df.columns:
        raise ValueError("Target column 'category' missing from dataset.")

    y = df["category"].copy()

    # Define strict exclusion list to prevent data and target leakage
    prohibited_columns = [
        "transaction_id",
        "description",         # Severe leakage: f"{category} expense"
        "category",            # Target variable
        "is_anomaly",          # Unrelated anomaly target
        "anomaly_score",       # Derived anomaly metric
        "currency",            # Constant ('INR')
        "is_business_expense", # Constant (True)
        "employee_id",         # High cardinality (6,246 unique IDs in 10k rows)
        "transaction_date",    # Raw string replaced by engineered calendar features
    ]

    feature_cols = [c for c in df.columns if c not in prohibited_columns]
    X = df[feature_cols].copy()

    # Identify numeric and categorical columns
    numeric_features = [c for c in ["amount", "log_amount", "vendor_rating", "month", "day"] if c in X.columns]
    categorical_features = [c for c in ["payment_method", "transaction_type", "department_id", "vendor_category", "day_of_week", "is_weekend"] if c in X.columns]

    # Convert categorical integers to string so OneHotEncoder handles them cleanly
    for col in categorical_features:
        X[col] = X[col].astype(str)

    # Ensure no prohibited columns leaked into X
    leaked = set(X.columns).intersection(set(prohibited_columns))
    if leaked:
        raise RuntimeError(f"LEAKAGE DETECTED! Prohibited columns present in classification features: {leaked}")

    return X, y, numeric_features, categorical_features


def get_anomaly_features(
    df: pd.DataFrame,
    category_medians: Optional[Dict[str, float]] = None,
    department_medians: Optional[Dict[Any, float]] = None
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str], Dict[str, float], Dict[Any, float]]:
    """
    Extracts features and target for Model B: Anomaly Detection.

    STRICT LEAKAGE GUARD:
    - Excludes 'is_anomaly' and 'anomaly_score' from X
    - Excludes 'transaction_id'
    - Excludes 'description'
    - Excludes constant columns ('currency', 'is_business_expense') and 'employee_id'

    Features include:
    - Amount & log-transformed amount
    - Ratio of transaction amount to category median spending
    - Ratio of transaction amount to department median spending
    - Day-of-week and weekend indicators (weekend was a generative anomaly trigger)
    - Vendor risk level (HIGH was a generative anomaly trigger)
    - Payment method, transaction type, department, and category

    Returns:
        X: Feature DataFrame
        y: Ground truth binary target Series (is_anomaly: 0 or 1)
        numeric_features: List of numeric column names in X
        categorical_features: List of categorical column names in X
        category_medians: Dict of median amounts per category
        department_medians: Dict of median amounts per department
    """
    if "is_anomaly" not in df.columns:
        raise ValueError("Target column 'is_anomaly' missing from dataset.")

    y = df["is_anomaly"].astype(int).copy()

    # Calculate or apply training-set medians to prevent test-set leakage
    if category_medians is None:
        cat_medians = df.groupby("category")["amount"].median().to_dict()
    else:
        cat_medians = category_medians

    if department_medians is None:
        dept_medians = df.groupby("department_id")["amount"].median().to_dict()
    else:
        dept_medians = department_medians

    # Compute comparative financial features
    df_feat = df.copy()
    cat_med_series = df_feat["category"].map(cat_medians).fillna(df_feat["amount"].median())
    dept_med_series = df_feat["department_id"].map(dept_medians).fillna(df_feat["amount"].median())

    df_feat["amount_to_category_median"] = df_feat["amount"] / np.maximum(cat_med_series, 1.0)
    df_feat["amount_to_dept_median"] = df_feat["amount"] / np.maximum(dept_med_series, 1.0)

    prohibited_columns = [
        "transaction_id",
        "is_anomaly",          # Target variable
        "anomaly_score",       # Direct target leakage
        "description",         # Excluded for consistency
        "currency",
        "is_business_expense",
        "employee_id",
        "transaction_date",
    ]

    feature_cols = [c for c in df_feat.columns if c not in prohibited_columns]
    X = df_feat[feature_cols].copy()

    numeric_features = [
        c for c in [
            "amount",
            "log_amount",
            "amount_to_category_median",
            "amount_to_dept_median",
            "vendor_rating",
            "month",
            "day"
        ] if c in X.columns
    ]

    categorical_features = [
        c for c in [
            "category",          # Category is a valid context feature for anomaly detection
            "payment_method",
            "transaction_type",
            "department_id",
            "risk_level",
            "vendor_category",
            "day_of_week",
            "is_weekend"
        ] if c in X.columns
    ]

    for col in categorical_features:
        X[col] = X[col].astype(str)

    # Ensure no prohibited columns leaked into X
    leaked = set(X.columns).intersection(set(prohibited_columns))
    if leaked:
        raise RuntimeError(f"LEAKAGE DETECTED! Prohibited columns present in anomaly features: {leaked}")

    return X, y, numeric_features, categorical_features, cat_medians, dept_medians
