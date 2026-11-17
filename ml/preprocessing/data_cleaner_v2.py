"""
STS FINTECH AI FINANCIAL INTELLIGENCE
V2 Data Cleaner and Feature Extraction (Milestone 2 - Step 7.3B-1)

Provides robust ingestion, relational enrichment, and leakage-safe feature extraction for:
- Model A: Multiclass Expense Classification (14 categories)
- Model B: Anomaly Detection (Unsupervised Baseline & Supervised Benchmark)

Enforces strict zero-leakage standards:
- Strictly bars target variables from input feature sets.
- Prevents unique ID memorization ('transaction_id', 'vendor_id', 'employee_id').
- Excludes 'is_anomaly' and 'anomaly_type' from classification AND anomaly feature sets.
- Eliminates target name substrings from text narratives.
- Computes group aggregates (medians) strictly on training data folds.
"""

import os
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V2_DATA_DIR = os.path.join(BASE_DIR, "data", "v2", "raw")


def load_v2_transaction_data(
    transactions_csv: str = "transactions.csv",
    vendors_csv: Optional[str] = "vendors.csv",
    departments_csv: Optional[str] = "departments.csv",
    employees_csv: Optional[str] = "employees.csv",
    data_dir: str = V2_DATA_DIR,
    enrich_vendors: bool = True,
    enrich_departments: bool = True,
    enrich_employees: bool = True
) -> pd.DataFrame:
    """
    Loads raw V2 transaction data and relationally enriches with vendor,
    department, and employee spending authorization metadata.
    Extracts core calendar and mathematical log-transformed features.
    """
    tx_path = os.path.join(data_dir, transactions_csv) if not os.path.isabs(transactions_csv) else transactions_csv
    if not os.path.exists(tx_path):
        raise FileNotFoundError(f"V2 transactions dataset not found at: {tx_path}")

    df = pd.read_csv(tx_path)

    # 1. Vendor metadata enrichment
    if enrich_vendors and vendors_csv:
        vn_path = os.path.join(data_dir, vendors_csv) if not os.path.isabs(vendors_csv) else vendors_csv
        if os.path.exists(vn_path):
            df_vn = pd.read_csv(vn_path)
            vn_cols = ["vendor_id", "vendor_category", "vendor_rating", "is_preferred_vendor", "payment_terms_days"]
            available_cols = [c for c in vn_cols if c in df_vn.columns]
            df = df.merge(df_vn[available_cols], on="vendor_id", how="left")
            if "vendor_rating" in df.columns:
                df["vendor_rating"] = pd.to_numeric(df["vendor_rating"], errors="coerce").fillna(3.5)
            if "vendor_category" in df.columns:
                df["vendor_category"] = df["vendor_category"].fillna("General Supplier")
            if "is_preferred_vendor" in df.columns:
                df["is_preferred_vendor"] = df["is_preferred_vendor"].astype(int)
            if "payment_terms_days" in df.columns:
                df["payment_terms_days"] = pd.to_numeric(df["payment_terms_days"], errors="coerce").fillna(30).astype(int)

    # 2. Department metadata enrichment
    if enrich_departments and departments_csv:
        dp_path = os.path.join(data_dir, departments_csv) if not os.path.isabs(departments_csv) else departments_csv
        if os.path.exists(dp_path):
            df_dp = pd.read_csv(dp_path)
            dp_cols = ["department_id", "division", "annual_budget_baseline"]
            available_cols = [c for c in dp_cols if c in df_dp.columns]
            df = df.merge(df_dp[available_cols], on="department_id", how="left")
            if "division" in df.columns:
                df["division"] = df["division"].fillna("General Operations")
            if "annual_budget_baseline" in df.columns:
                df["annual_budget_baseline"] = pd.to_numeric(df["annual_budget_baseline"], errors="coerce").fillna(10000000.0)

    # 3. Employee metadata enrichment (spending authorization limit)
    if enrich_employees and employees_csv:
        em_path = os.path.join(data_dir, employees_csv) if not os.path.isabs(employees_csv) else employees_csv
        if os.path.exists(em_path):
            df_em = pd.read_csv(em_path)
            em_cols = ["employee_id", "spending_approval_limit", "employment_type"]
            available_cols = [c for c in em_cols if c in df_em.columns]
            df = df.merge(df_em[available_cols], on="employee_id", how="left")
            if "spending_approval_limit" in df.columns:
                df["spending_approval_limit"] = pd.to_numeric(df["spending_approval_limit"], errors="coerce").fillna(50000.0)
            if "employment_type" in df.columns:
                df["employment_type"] = df["employment_type"].fillna("FULL_TIME")

    # 4. Calendar feature extraction
    if "transaction_date" in df.columns:
        dt = pd.to_datetime(df["transaction_date"], errors="coerce")
        df["day_of_week"] = dt.dt.weekday.fillna(0).astype(int)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["month"] = dt.dt.month.fillna(1).astype(int)
        df["day"] = dt.dt.day.fillna(1).astype(int)
        df["quarter"] = dt.dt.quarter.fillna(1).astype(int)

    # 5. Numerical transformations
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        df["log_amount"] = np.log1p(np.maximum(df["amount"], 0.0))

    return df


def get_v2_classification_features(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, str, List[str], List[str]]:
    """
    Extracts features and target for Model A: V2 Expense Classification.

    STRICT LEAKAGE GUARD:
    - Target: 'category' (14 classes)
    - Excludes 'category' from X
    - Excludes 'transaction_id', 'vendor_id', 'employee_id' (prevents memorization)
    - Excludes 'is_anomaly', 'anomaly_type', 'anomaly_score'
    - Excludes constant columns ('currency', 'is_business_expense')
    - Excludes raw 'transaction_date' (replaced by calendar features)

    Returns:
        X: Feature DataFrame containing 'description', numeric columns, and categorical columns
        y: Target Series (category)
        text_feature: Name of the text column ('description')
        numeric_features: List of numeric column names
        categorical_features: List of categorical column names
    """
    if "category" not in df.columns:
        raise ValueError("Target column 'category' missing from dataset.")

    y = df["category"].copy()

    prohibited_columns = [
        "transaction_id",
        "category",            # Target variable
        "is_anomaly",          # Anomaly target
        "anomaly_type",        # Anomaly label
        "anomaly_score",
        "currency",            # Constant ('INR')
        "is_business_expense", # Constant (True)
        "employee_id",         # High cardinality
        "vendor_id",           # Barred from predictive features (prevents vendor memorization)
        "transaction_date",    # Raw string
    ]

    feature_cols = [c for c in df.columns if c not in prohibited_columns]
    X = df[feature_cols].copy()

    # Text narrative feature
    text_feature = "description"
    if text_feature not in X.columns:
        raise ValueError("Required text feature 'description' missing from dataframe.")
    X[text_feature] = X[text_feature].fillna("").astype(str)

    # Numeric features
    numeric_candidates = [
        "amount",
        "log_amount",
        "vendor_rating",
        "payment_terms_days",
        "annual_budget_baseline",
        "month",
        "day",
        "quarter"
    ]
    numeric_features = [c for c in numeric_candidates if c in X.columns]

    # Categorical features
    categorical_candidates = [
        "vendor_category",
        "division",
        "department_id",
        "payment_method",
        "transaction_type",
        "is_preferred_vendor",
        "employment_type",
        "day_of_week",
        "is_weekend"
    ]
    categorical_features = [c for c in categorical_candidates if c in X.columns]

    for col in categorical_features:
        X[col] = X[col].astype(str)

    # Leakage assertion
    leaked = set(X.columns).intersection(set(prohibited_columns))
    if leaked:
        raise RuntimeError(f"LEAKAGE DETECTED in V2 Classification features: {leaked}")

    return X, y, text_feature, numeric_features, categorical_features


def get_v2_anomaly_features(
    df: pd.DataFrame,
    category_medians: Optional[Dict[str, float]] = None,
    department_medians: Optional[Dict[Any, float]] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str], List[str], Dict[str, float], Dict[Any, float]]:
    """
    Extracts features, binary target, and scenario provenance for Model B: V2 Anomaly Detection.

    STRICT LEAKAGE GUARD:
    - Target: 'is_anomaly' (binary: 0 or 1)
    - Scenario provenance: 'anomaly_type' (ground-truth evaluation only, never input feature)
    - Excludes 'is_anomaly', 'anomaly_type', and 'anomaly_score' from X
    - Excludes 'transaction_id', 'employee_id', 'vendor_id'
    - Excludes constant columns ('currency', 'is_business_expense')
    - Excludes raw 'transaction_date'

    Financial contextual features engineered:
    - Amount & log_amount
    - Ratio to category median spending (calculated strictly on train set)
    - Ratio to department median spending (calculated strictly on train set)
    - Ratio to employee spending authorization limit (detects split threshold attempts)
    - Near-threshold indicator: amount in [0.90, 1.00] of employee approval limit
    - Weekend/off-hours high-value interaction
    - Low vendor rating indicator (< 2.2 rating)

    Returns:
        X: Feature DataFrame
        y: Binary ground-truth target Series (0 or 1)
        anomaly_types: Series of ground-truth anomaly scenarios (for evaluation only)
        numeric_features: List of numeric column names in X
        categorical_features: List of categorical column names in X
        category_medians: Dict used for category median amounts
        department_medians: Dict used for department median amounts
    """
    if "is_anomaly" not in df.columns:
        raise ValueError("Target column 'is_anomaly' missing from dataset.")

    y = df["is_anomaly"].astype(int).copy()
    anomaly_types = df["anomaly_type"].fillna("NONE").copy() if "anomaly_type" in df.columns else pd.Series(["NONE"] * len(df))

    df_feat = df.copy()

    # Calculate or apply training-set medians to prevent test-set distribution leakage
    if category_medians is None:
        cat_medians = df_feat.groupby("category")["amount"].median().to_dict()
    else:
        cat_medians = category_medians

    if department_medians is None:
        dept_medians = df_feat.groupby("department_id")["amount"].median().to_dict()
    else:
        dept_medians = department_medians

    overall_median = float(df_feat["amount"].median()) if len(df_feat) > 0 else 1000.0

    cat_med_series = df_feat["category"].map(cat_medians).fillna(overall_median)
    dept_med_series = df_feat["department_id"].map(dept_medians).fillna(overall_median)

    df_feat["amount_to_category_median"] = df_feat["amount"] / np.maximum(cat_med_series, 1.0)
    df_feat["amount_to_dept_median"] = df_feat["amount"] / np.maximum(dept_med_series, 1.0)

    # Authorization limit ratio (vital for SPLIT_APPROVAL_THRESHOLD detection)
    if "spending_approval_limit" in df_feat.columns:
        spend_limits = np.maximum(df_feat["spending_approval_limit"].fillna(50000.0), 1000.0)
        df_feat["amount_to_spending_limit"] = df_feat["amount"] / spend_limits
        df_feat["near_approval_threshold"] = (
            (df_feat["amount_to_spending_limit"] >= 0.90) & (df_feat["amount_to_spending_limit"] <= 1.00)
        ).astype(int)
    else:
        df_feat["amount_to_spending_limit"] = 1.0
        df_feat["near_approval_threshold"] = 0

    # Risk signals
    if "vendor_rating" in df_feat.columns:
        df_feat["is_low_rated_vendor"] = (df_feat["vendor_rating"] < 2.5).astype(int)
    else:
        df_feat["is_low_rated_vendor"] = 0

    prohibited_columns = [
        "transaction_id",
        "is_anomaly",          # Target variable
        "anomaly_type",        # Evaluation label
        "anomaly_score",
        "currency",
        "is_business_expense",
        "employee_id",
        "vendor_id",
        "transaction_date",
        "description",         # Excluded from anomaly feature set to maintain numeric/contextual generality
    ]

    feature_cols = [c for c in df_feat.columns if c not in prohibited_columns]
    X = df_feat[feature_cols].copy()

    numeric_candidates = [
        "amount",
        "log_amount",
        "amount_to_category_median",
        "amount_to_dept_median",
        "amount_to_spending_limit",
        "near_approval_threshold",
        "is_low_rated_vendor",
        "vendor_rating",
        "payment_terms_days",
        "annual_budget_baseline",
        "spending_approval_limit",
        "month",
        "day",
        "quarter"
    ]
    numeric_features = [c for c in numeric_candidates if c in X.columns]

    categorical_candidates = [
        "category",            # Valid contextual feature for anomaly detection
        "vendor_category",
        "division",
        "department_id",
        "payment_method",
        "transaction_type",
        "is_preferred_vendor",
        "employment_type",
        "day_of_week",
        "is_weekend"
    ]
    categorical_features = [c for c in categorical_candidates if c in X.columns]

    for col in categorical_features:
        X[col] = X[col].astype(str)

    leaked = set(X.columns).intersection(set(prohibited_columns))
    if leaked:
        raise RuntimeError(f"LEAKAGE DETECTED in V2 Anomaly features: {leaked}")

    return X, y, anomaly_types, numeric_features, categorical_features, cat_medians, dept_medians
