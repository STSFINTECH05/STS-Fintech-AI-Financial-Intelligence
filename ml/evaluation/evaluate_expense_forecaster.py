"""
STS FINTECH AI FINANCIAL INTELLIGENCE
Expense Forecaster Comprehensive Evaluation & Projection Runner (Milestone 2 - Step 7.2C)

Coordinates Model C training, full validation & test evaluations,
generates multi-horizon forecasts (7, 14, 30 days), and prints
detailed comparison tables against Naive and Ridge baselines.
"""

import os
import sys
import json
import hashlib
from typing import Dict, Any
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from ml.training.train_expense_forecaster import train_expense_forecaster
from ml.prediction.expense_forecaster import ExpenseForecaster


def compute_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_full_forecaster_evaluation():
    print("=" * 70)
    print("STS FINTECH - MILESTONE 2 - STEP 7.2C EXPENSE FORECASTER EVALUATION")
    print("=" * 70)

    # 1. Execute Training & Base Evaluation Pipeline
    model, metadata = train_expense_forecaster()

    # 2. Initialize Inference Forecaster
    forecaster = ExpenseForecaster()

    # 3. Generate Multi-Horizon Forecasts (7, 14, 30 days)
    print("\n" + "=" * 70)
    print("RECURSIVE MULTI-HORIZON EXPENSE PROJECTIONS (Post 2026-10-31)")
    print("=" * 70)

    for horizon in [7, 14, 30]:
        fc = forecaster.forecast(horizon_days=horizon)
        print(f"\n--- {horizon}-Day Forecast Summary ---")
        print(f"  As of Cutoff:      {fc['as_of_date']}")
        print(f"  Total Expenditure: ₹{fc['total_projected_expenditure']:,.2f}")
        print(f"  Daily Mean:        ₹{fc['average_daily_projected']:,.2f}")
        print(f"  First 3 Days:")
        for p in fc["projections"][:3]:
            print(f"    {p['forecast_date']} ({p['day_name']}): ₹{p['predicted_expenditure']:,.2f} [{p['projection_status']}]")
        print(f"  Last Day:")
        last = fc["projections"][-1]
        print(f"    {last['forecast_date']} ({last['day_name']}): ₹{last['predicted_expenditure']:,.2f} [{last['projection_status']}]")

    # 4. Verify artifact on disk and checksum
    model_path = os.path.join(BASE_DIR, "ml", "models", "expense_forecaster.joblib")
    checksum = compute_sha256(model_path)
    metadata["sha256_checksum"] = checksum

    meta_path = os.path.join(BASE_DIR, "ml", "models", "expense_forecaster_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ All artifacts and metadata successfully verified on disk:")
    print(f"  Model Artifact: {model_path} ({os.path.getsize(model_path):,} bytes)")
    print(f"  SHA-256 Checksum: {checksum}")
    print(f"  Metadata:       {meta_path}")

    return metadata


if __name__ == "__main__":
    run_full_forecaster_evaluation()
