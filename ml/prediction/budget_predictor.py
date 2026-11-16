"""
STS Fintech – Milestone 2 – Step 7.2D: Budget Prediction Inference Module
Predicts departmental budget utilization, final expenditures, remaining allocations,
and overspending risk tiers with strict input validation.
"""

import os
import calendar
import numpy as np
import pandas as pd
import joblib
from scipy.stats import norm


MODEL_PATH = "ml/models/budget_predictor.joblib"


class BudgetPredictor:
    """
    Production-ready inference engine for departmental budget utilization forecasting.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Budget predictor model artifact not found at: {model_path}")

        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.scaler = bundle.get("scaler")
        self.uses_scaler = bundle.get("uses_scaler", False)
        self.model_name = bundle.get("model_name", "BudgetPredictor")
        self.feature_names = bundle.get("feature_names", [])
        self.department_profiles = bundle.get("department_profiles", {})
        self.residual_std = bundle.get("residual_std", 11.4)

    def predict(
        self,
        department_id: int,
        budget_year: int,
        budget_month: int,
        allocated_amount: float,
        line_count: int = None,
        hist_mean_util: float = None,
        hist_overrun_freq: float = None,
        log_hist_mean_alloc: float = None,
        lag_1_util: float = None,
        lag_2_util: float = None,
        rolling_3m_util: float = None,
        month_to_date_spend: float = None,
        as_of_day: int = None
    ) -> dict:
        """
        Predicts expected monthly budget utilization and overspending risk.

        Args:
            department_id: Department ID (1 to 50).
            budget_year: Calendar year (e.g., 2025, 2026).
            budget_month: Month of year (1 to 12).
            allocated_amount: Total allocated budget in INR (> 0).
            line_count: Count of allocation lines for the month.
            hist_mean_util: Optional override for department historical mean utilization.
            hist_overrun_freq: Optional override for historical overrun frequency.
            log_hist_mean_alloc: Optional override for historical mean allocated budget (log).
            lag_1_util: Optional prior month utilization.
            lag_2_util: Optional 2-month prior utilization.
            rolling_3m_util: Optional rolling 3-month mean utilization.
            month_to_date_spend: Optional actual spend recorded month-to-date.
            as_of_day: Optional day of month (1 to 31) corresponding to MTD spend.

        Returns:
            Structured dictionary with predicted utilization, expenditure, and risk tier.
        """
        # Input Validation
        if allocated_amount is None or allocated_amount <= 0:
            raise ValueError("allocated_amount must be a strictly positive number.")
        if budget_month < 1 or budget_month > 12:
            raise ValueError(f"budget_month must be between 1 and 12, got: {budget_month}")
        if department_id not in self.department_profiles and (department_id < 1 or department_id > 100):
            raise ValueError(f"Unknown or invalid department_id: {department_id}")

        dept_id = int(department_id)
        profile = self.department_profiles.get(dept_id, {
            "hist_mean_util": 84.6,
            "hist_overrun_freq": 0.10,
            "hist_mean_alloc": allocated_amount,
            "lag_1_util": 84.6,
            "lag_2_util": 84.6,
            "rolling_3m_util": 84.6,
            "typical_line_count": 4
        })

        # Populate features with fallback to historical department profiles
        h_util = hist_mean_util if hist_mean_util is not None else profile["hist_mean_util"]
        h_over = hist_overrun_freq if hist_overrun_freq is not None else profile["hist_overrun_freq"]
        h_alloc = log_hist_mean_alloc if log_hist_mean_alloc is not None else np.log1p(profile["hist_mean_alloc"])
        l1 = lag_1_util if lag_1_util is not None else profile["lag_1_util"]
        l2 = lag_2_util if lag_2_util is not None else profile["lag_2_util"]
        r3 = rolling_3m_util if rolling_3m_util is not None else profile["rolling_3m_util"]
        l_cnt = line_count if line_count is not None else profile["typical_line_count"]

        quarter = (budget_month - 1) // 3 + 1
        log_alloc = float(np.log1p(max(0.0, allocated_amount)))

        features = np.array([[
            dept_id,
            budget_year,
            budget_month,
            quarter,
            log_alloc,
            l_cnt,
            h_util,
            h_over,
            h_alloc,
            l1,
            l2,
            r3
        ]], dtype=np.float64)

        if self.uses_scaler and self.scaler is not None:
            features = self.scaler.transform(features)

        pred_util = float(self.model.predict(features)[0])
        pred_util = float(np.clip(pred_util, 0.0, 200.0))

        expected_expenditure = round(allocated_amount * (pred_util / 100.0), 2)
        estimated_remaining = round(allocated_amount - expected_expenditure, 2)

        # Standardized Risk Tiers
        if pred_util < 80.0:
            risk_tier = "Within Budget"
        elif pred_util <= 100.0:
            risk_tier = "Approaching Budget Limit"
        else:
            risk_tier = "Potential Budget Overrun"

        # Overrun probability using Gaussian tail
        z_score = (100.0 - pred_util) / max(1.0, self.residual_std)
        overrun_prob = float(norm.sf(z_score))

        result = {
            "department_id": dept_id,
            "budget_year": budget_year,
            "budget_month": budget_month,
            "allocated_amount": round(float(allocated_amount), 2),
            "expected_final_utilization_pct": round(pred_util, 2),
            "expected_final_expenditure": expected_expenditure,
            "estimated_remaining_budget": estimated_remaining,
            "predicted_overspending_risk_tier": risk_tier,
            "overspending_risk_probability": round(overrun_prob, 4),
            "model_metadata": {
                "algorithm": self.model_name,
                "historical_department_mean_util": round(h_util, 2),
                "historical_department_overrun_freq": round(h_over, 4)
            }
        }

        # Optional intra-month run-rate analysis
        if month_to_date_spend is not None and as_of_day is not None:
            days_in_month = calendar.monthrange(budget_year, budget_month)[1]
            valid_day = min(max(1, as_of_day), days_in_month)
            elapsed_pct = valid_day / days_in_month
            projected_mtd_runrate = (month_to_date_spend / elapsed_pct) if elapsed_pct > 0 else month_to_date_spend
            mtd_utilization_pace = (projected_mtd_runrate / allocated_amount) * 100.0

            result["intra_month_snapshot"] = {
                "as_of_day": valid_day,
                "days_in_month": days_in_month,
                "elapsed_month_pct": round(elapsed_pct * 100.0, 1),
                "month_to_date_spend": round(float(month_to_date_spend), 2),
                "projected_mtd_full_month_runrate": round(float(projected_mtd_runrate), 2),
                "mtd_utilization_pace_pct": round(float(mtd_utilization_pace), 2),
                "note": "Intra-month run-rate calculated from user-provided verified snapshot. Not inferred from cross-table transactions."
            }

        return result
