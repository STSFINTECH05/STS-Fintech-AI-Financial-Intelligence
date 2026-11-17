"""
STS Fintech – Milestone 2 – Step 7.3B-3
Model D: V2 Budget Utilization Prediction Inference Module

Provides production-ready inference for departmental budget utilization forecasting,
final monthly expenditure estimation, remaining allocations, and overrun risk screening.
"""

import os
import calendar
import numpy as np
import pandas as pd
import joblib
from scipy.stats import norm

DEFAULT_MODEL_PATH = "ml/models/v2/budget_predictor_v2.joblib"


class BudgetPredictorV2:
    """
    Production-grade inference engine for V2 departmental budget utilization forecasting.
    Predicts final monthly actual spend, utilization percentage, remaining allocation,
    and classifies budget overrun risk into standard enterprise advisory tiers.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"V2 Budget predictor model artifact not found at: {model_path}")

        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.model_name = bundle.get("model_name", "RandomForestRegressor")
        self.feature_names = bundle.get("feature_names", [])
        self.department_profiles = bundle.get("department_profiles", {})
        self.residual_std = bundle.get("residual_std", 11.5)
        self.overall_metrics = bundle.get("overall_test_metrics", {})
        self.day_metrics = bundle.get("day_metrics", {})
        self.baselines = bundle.get("baselines", {})

    def predict(
        self,
        department_id: int,
        budget_year: int,
        budget_month: int,
        as_of_day: int,
        allocated_amount: float,
        month_to_date_actual: float,
        transaction_count_mtd: int = None,
        annual_budget_baseline: float = None
    ) -> dict:
        """
        Predicts final monthly expenditure, utilization %, and overrun risk for a department.

        Args:
            department_id: Department ID (1 to 50 or registered).
            budget_year: Calendar year (e.g., 2026).
            budget_month: Month of year (1 to 12).
            as_of_day: Current snapshot day of the month (1 to 31).
            allocated_amount: Allocated budget in INR (must be > 0).
            month_to_date_actual: Spend recorded month-to-date in INR (>= 0).
            transaction_count_mtd: Number of transactions recorded month-to-date.
            annual_budget_baseline: Department annual baseline budget (INR).

        Returns:
            Dictionary containing predicted utilization, final amount, remaining budget,
            overrun risk classification, and linear run-rate comparison.
        """
        # Input Validation
        if allocated_amount is None or allocated_amount <= 0:
            raise ValueError(f"allocated_amount must be strictly positive, got: {allocated_amount}")
        if month_to_date_actual is None or month_to_date_actual < 0:
            raise ValueError(f"month_to_date_actual must be non-negative, got: {month_to_date_actual}")
        if budget_month < 1 or budget_month > 12:
            raise ValueError(f"budget_month must be between 1 and 12, got: {budget_month}")

        days_in_month = calendar.monthrange(int(budget_year), int(budget_month))[1]
        if as_of_day < 1 or as_of_day > days_in_month:
            raise ValueError(f"as_of_day must be between 1 and {days_in_month} for month {budget_month}, got: {as_of_day}")

        dept_id = int(department_id)
        dept_profile = self.department_profiles.get(dept_id, {
            "hist_mean_util": 85.0,
            "hist_median_util": 85.0,
            "hist_overrun_freq": 0.15,
            "annual_budget_baseline": allocated_amount * 12.0
        })

        if annual_budget_baseline is None or annual_budget_baseline <= 0:
            annual_budget_baseline = dept_profile.get("annual_budget_baseline", allocated_amount * 12.0)

        # Elapsed and Remaining Days
        elapsed_pct = (as_of_day / days_in_month) * 100.0
        remaining_days = days_in_month - as_of_day

        # Fallback for transaction count: estimate from average daily spend
        if transaction_count_mtd is None or transaction_count_mtd < 0:
            # Estimate ~1 transaction per 15,000 INR
            transaction_count_mtd = max(1, int(round(month_to_date_actual / 15000.0)))

        # Feature Construction
        linear_runrate = month_to_date_actual / max(1e-4, (elapsed_pct / 100.0))
        mtd_util_pct = (month_to_date_actual / allocated_amount) * 100.0
        linear_util_pct = (linear_runrate / allocated_amount) * 100.0
        tx_rate_per_day = transaction_count_mtd / max(1, as_of_day)
        avg_tx_size = month_to_date_actual / max(1, transaction_count_mtd)
        alloc_vs_baseline = allocated_amount / max(1.0, (annual_budget_baseline / 12.0))
        remaining_budget_mtd = allocated_amount - month_to_date_actual

        feature_row = {
            "department_id": dept_id,
            "budget_month": int(budget_month),
            "as_of_day": int(as_of_day),
            "elapsed_month_pct": float(elapsed_pct),
            "allocated_amount": float(allocated_amount),
            "month_to_date_actual": float(month_to_date_actual),
            "transaction_count_mtd": int(transaction_count_mtd),
            "days_in_month": int(days_in_month),
            "remaining_days": int(remaining_days),
            "linear_runrate": float(linear_runrate),
            "mtd_utilization_pct": float(mtd_util_pct),
            "linear_runrate_utilization_pct": float(linear_util_pct),
            "tx_rate_per_day": float(tx_rate_per_day),
            "avg_tx_size_mtd": float(avg_tx_size),
            "alloc_vs_baseline": float(alloc_vs_baseline),
            "remaining_budget_mtd": float(remaining_budget_mtd)
        }

        X_df = pd.DataFrame([feature_row])[self.feature_names]

        # ML Model Prediction
        pred_util = float(self.model.predict(X_df)[0])
        # Non-negative constraint and reasonable bounds (utilization cannot be lower than current MTD util)
        pred_util = max(mtd_util_pct, max(0.0, pred_util))
        pred_actual = float(round(allocated_amount * pred_util / 100.0, 2))
        est_remaining = float(round(allocated_amount - pred_actual, 2))

        # Overrun Probability via Normal Distribution of Residuals
        # P(util > 100%) = 1 - Phi((100 - pred_util) / residual_std)
        z_score = (100.0 - pred_util) / max(0.1, self.residual_std)
        overrun_prob = float(np.clip(1.0 - norm.cdf(z_score), 0.0, 1.0))

        # Advisory Risk Tier
        if pred_util > 100.0:
            advisory_tier = "Potential Budget Overrun – Requires Verification"
            action = "Immediate expenditure freeze or formal budget reallocation request required."
        elif pred_util >= 90.0:
            advisory_tier = "Approaching Budget Limit"
            action = "Close monitoring of pending purchase orders; enforce approval controls."
        else:
            advisory_tier = "Within Budget"
            action = "Standard procurement operations; spending is tracking within healthy parameters."

        return {
            "department_id": dept_id,
            "budget_year": int(budget_year),
            "budget_month": int(budget_month),
            "as_of_day": int(as_of_day),
            "allocated_amount": float(allocated_amount),
            "month_to_date_actual": float(month_to_date_actual),
            "predicted_utilization_pct": round(pred_util, 2),
            "predicted_final_actual": pred_actual,
            "estimated_remaining_budget": est_remaining,
            "potential_overrun": bool(pred_actual > allocated_amount),
            "overrun_probability": round(overrun_prob, 4),
            "advisory_risk_tier": advisory_tier,
            "recommended_action": action,
            "baselines": {
                "linear_runrate_actual": round(linear_runrate, 2),
                "linear_runrate_utilization_pct": round(linear_util_pct, 2),
                "dept_historical_median_utilization": dept_profile.get("hist_median_util", 85.0)
            },
            "model_metadata": {
                "model_name": self.model_name,
                "residual_std": round(self.residual_std, 2)
            }
        }

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch prediction for a dataframe of budget snapshots.
        """
        results = []
        for _, row in df.iterrows():
            res = self.predict(
                department_id=int(row["department_id"]),
                budget_year=int(row["budget_year"]),
                budget_month=int(row["budget_month"]),
                as_of_day=int(row["as_of_day"]),
                allocated_amount=float(row["allocated_amount"]),
                month_to_date_actual=float(row["month_to_date_actual"]),
                transaction_count_mtd=int(row.get("transaction_count_mtd", 0)),
                annual_budget_baseline=float(row.get("annual_budget_baseline", 0)) if "annual_budget_baseline" in row else None
            )
            results.append({
                "predicted_utilization_pct": res["predicted_utilization_pct"],
                "predicted_final_actual": res["predicted_final_actual"],
                "estimated_remaining_budget": res["estimated_remaining_budget"],
                "potential_overrun": res["potential_overrun"],
                "overrun_probability": res["overrun_probability"],
                "advisory_risk_tier": res["advisory_risk_tier"]
            })
        return pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
