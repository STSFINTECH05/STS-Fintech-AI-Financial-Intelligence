"""
STS Fintech – Milestone 2 – Step 7.3B-3
Model E: V2 Vendor Intelligence Inference Module

Unifies:
1. Duplicate invoice screening (operational deterministic matcher)
2. Accounts payable late payment prediction
3. Multi-factor vendor risk screening and review prioritization
"""

import os
import joblib
import numpy as np
import pandas as pd

DEFAULT_DUP_PATH = "ml/models/v2/vendor_duplicate_detector_v2.joblib"
DEFAULT_PAY_PATH = "ml/models/v2/vendor_payment_predictor_v2.joblib"
DEFAULT_RISK_PATH = "ml/models/v2/vendor_screening_v2.joblib"


class OperationalDuplicateMatcher:
    """
    Enterprise deterministic rule engine that screens invoices against historical
    registries for exact and near-match duplicates (same vendor, identical/near amount,
    within configurable day window).
    """

    def __init__(self, window_days: int = 30, amount_tolerance_pct: float = 0.01):
        self.window_days = window_days
        self.amount_tolerance_pct = amount_tolerance_pct
        self.registry = []  # list of dicts: {invoice_id, vendor_id, total_amount, invoice_date}

    def register_invoices(self, invoices_df: pd.DataFrame):
        """Populates the known invoice registry."""
        for _, row in invoices_df.iterrows():
            self.registry.append({
                "invoice_id": str(row["invoice_id"]),
                "vendor_id": str(row["vendor_id"]),
                "total_amount": float(row["total_amount"]),
                "invoice_date": pd.to_datetime(row["invoice_date"])
            })

    def check_duplicate(self, vendor_id: str, total_amount: float, invoice_date: str) -> dict:
        """
        Screens an incoming invoice candidate against registered invoices.
        """
        target_vid = str(vendor_id).strip().upper()
        target_amt = float(total_amount)
        target_date = pd.to_datetime(invoice_date)

        matches = []
        for inv in self.registry:
            if inv["vendor_id"] == target_vid:
                amt_diff_pct = abs(inv["total_amount"] - target_amt) / max(1.0, target_amt)
                date_diff_days = abs((inv["invoice_date"] - target_date).days)

                if date_diff_days <= self.window_days:
                    if amt_diff_pct <= self.amount_tolerance_pct:
                        matches.append({
                            "matched_invoice_id": inv["invoice_id"],
                            "matched_amount": inv["total_amount"],
                            "matched_date": inv["invoice_date"].strftime("%Y-%m-%d"),
                            "day_delta": date_diff_days,
                            "match_type": "EXACT" if amt_diff_pct == 0.0 else "NEAR_MATCH"
                        })

        is_dup = len(matches) > 0
        return {
            "is_suspected_duplicate": is_dup,
            "match_count": len(matches),
            "matches": matches,
            "screening_rule": f"Same vendor, amount within ±{self.amount_tolerance_pct*100:.1f}%, within ±{self.window_days} days"
        }


class VendorAnalyzerV2:
    """
    Enterprise inference engine unifying vendor risk screening, duplicate invoice
    detection, and accounts payable late settlement forecasting.
    """

    def __init__(
        self,
        risk_model_path: str = DEFAULT_RISK_PATH,
        dup_model_path: str = DEFAULT_DUP_PATH,
        pay_model_path: str = DEFAULT_PAY_PATH
    ):
        # 1. Load Risk Screening Bundle
        if not os.path.exists(risk_model_path):
            raise FileNotFoundError(f"Vendor screening artifact not found at: {risk_model_path}")
        risk_bundle = joblib.load(risk_model_path)
        self.profiles = risk_bundle.get("profiles", {})
        self.scoring_weights = risk_bundle.get("scoring_weights", {})
        self.scoring_thresholds = risk_bundle.get("scoring_thresholds", {"low_threshold": 25.0, "medium_threshold": 50.0})
        self.tier_distribution = risk_bundle.get("tier_distribution", {})

        # 2. Load Duplicate Detector Bundle
        self.matcher = None
        self.dup_clf = None
        self.dup_feature_names = []
        if os.path.exists(dup_model_path):
            dup_bundle = joblib.load(dup_model_path)
            self.matcher = dup_bundle.get("operational_matcher")
            self.dup_clf = dup_bundle.get("supervised_benchmark")
            self.dup_feature_names = dup_bundle.get("feature_names", [])

        # 3. Load Late Payment Predictor Bundle
        self.pay_clf = None
        self.pay_feature_names = []
        self.vendor_hist_rates = {}
        self.global_prior_late_rate = 0.2096
        if os.path.exists(pay_model_path):
            pay_bundle = joblib.load(pay_model_path)
            self.pay_clf = pay_bundle.get("model")
            self.pay_feature_names = pay_bundle.get("feature_names", [])
            self.vendor_hist_rates = pay_bundle.get("vendor_historical_rates", {})
            self.global_prior_late_rate = pay_bundle.get("global_prior_late_rate", 0.2096)

    def analyze_vendor(self, vendor_id: str) -> dict:
        """
        Retrieves comprehensive financial activity, friction indicators, and risk score for a vendor.

        Args:
            vendor_id: Vendor identifier (e.g., 'VEN00001').

        Returns:
            Structured dictionary with profile, scores, and review recommendations.
        """
        if not vendor_id or not isinstance(vendor_id, str):
            raise ValueError(f"vendor_id must be a non-empty string, got: {vendor_id}")

        vid = vendor_id.strip().upper()
        if vid not in self.profiles:
            raise KeyError(f"Vendor ID '{vid}' not found in registered enterprise vendor registry.")

        p = self.profiles[vid]
        return {
            "vendor_id": p["vendor_id"],
            "vendor_name": p["vendor_name"],
            "vendor_category": p["vendor_category"],
            "location": p["location"],
            "vendor_rating": p["vendor_rating"],
            "payment_terms_days": p["payment_terms_days"],
            "financial_activity": {
                "total_invoice_amount": p["total_invoice_amt"],
                "total_bill_amount": p["total_bill_amt"],
                "total_transaction_amount": p["total_tx_amt"],
                "invoice_count": p["invoice_count"],
                "bills_count": p["bills_count"],
                "transaction_count": p["tx_count"],
                "accounting_note": "Invoice, bill, and transaction figures represent distinct recorded ledgers."
            },
            "friction_indicators": {
                "rejected_invoices_count": p["rejected_invoices"],
                "duplicate_invoices_count": p["duplicate_invoices"],
                "late_bills_count": p["late_bills"],
                "overdue_bills_count": p["overdue_bills"],
                "transaction_anomalies_count": p["anomaly_tx_count"]
            },
            "risk_evaluation": {
                "risk_score": p["risk_score"],
                "risk_tier": p["risk_tier"],
                "factor_scores": p["factor_scores"],
                "scoring_weights": self.scoring_weights
            },
            "recommended_action": p["recommended_action"],
            "compliance_disclaimer": "Statistical risk indicator for screening prioritization only. Does not constitute a determination of fraud, default, or contractual breach."
        }

    def get_high_risk_vendors(self) -> list:
        """
        Returns all registered vendors classified as HIGH – REQUIRES VERIFICATION.
        """
        high_risk = []
        for vid, p in self.profiles.items():
            if p["risk_tier"] == "HIGH – REQUIRES VERIFICATION":
                high_risk.append({
                    "vendor_id": vid,
                    "vendor_name": p["vendor_name"],
                    "risk_score": p["risk_score"],
                    "risk_tier": p["risk_tier"],
                    "rejected_invoices": p["rejected_invoices"],
                    "duplicate_invoices": p["duplicate_invoices"],
                    "late_bills": p["late_bills"],
                    "anomaly_tx_count": p["anomaly_tx_count"],
                    "recommended_action": p["recommended_action"]
                })
        return sorted(high_risk, key=lambda x: x["risk_score"], reverse=True)

    def screen_invoice_duplicate(
        self,
        vendor_id: str,
        total_amount: float,
        invoice_date: str
    ) -> dict:
        """
        Screens an incoming invoice for exact or near-match duplicates.

        Args:
            vendor_id: Vendor identifier.
            total_amount: Invoice total amount in INR.
            invoice_date: Invoice date string (YYYY-MM-DD).

        Returns:
            Dictionary with duplicate determination and matched historical invoices.
        """
        if self.matcher is None:
            raise RuntimeError("OperationalDuplicateMatcher is not initialized.")

        return self.matcher.check_duplicate(vendor_id, total_amount, invoice_date)

    def predict_late_payment(
        self,
        vendor_id: str,
        total_amount: float,
        bill_date: str,
        due_date: str,
        department_id: int = 1,
        tax_amount: float = None
    ) -> dict:
        """
        Predicts late payment risk for an accounts payable bill.

        Args:
            vendor_id: Vendor identifier.
            total_amount: Total bill amount in INR.
            bill_date: Bill issue date (YYYY-MM-DD).
            due_date: Bill due date (YYYY-MM-DD).
            department_id: Responsible department ID.
            tax_amount: Optional tax amount.

        Returns:
            Dictionary with predicted late payment probability and risk flag.
        """
        if self.pay_clf is None:
            raise RuntimeError("Late payment classifier is not initialized.")

        vid = str(vendor_id).strip().upper()
        b_dt = pd.to_datetime(bill_date)
        d_dt = pd.to_datetime(due_date)
        terms_days = (d_dt - b_dt).days

        if tax_amount is None:
            tax_amount = total_amount * 0.18
        subtotal = total_amount - tax_amount

        v_profile = self.profiles.get(vid, {})
        v_rating = v_profile.get("vendor_rating", 3.75)
        prior_rate = self.vendor_hist_rates.get(vid, self.global_prior_late_rate)

        feature_row = {
            "department_id": int(department_id),
            "bill_amount": float(subtotal),
            "tax_amount": float(tax_amount),
            "total_amount": float(total_amount),
            "log_amount": float(np.log1p(total_amount)),
            "tax_ratio": float(tax_amount / max(1.0, total_amount)),
            "terms_days": int(terms_days),
            "vendor_rating": float(v_rating),
            "vendor_prior_late_rate": float(prior_rate),
            "month": int(b_dt.month),
            "day_of_week": int(b_dt.dayofweek),
            "day_of_month": int(b_dt.day)
        }

        X_df = pd.DataFrame([feature_row])[self.pay_feature_names]
        prob_late = float(self.pay_clf.predict_proba(X_df)[0, 1])
        is_suspected_late = bool(prob_late >= 0.50)

        return {
            "vendor_id": vid,
            "bill_date": bill_date,
            "due_date": due_date,
            "terms_days": terms_days,
            "total_amount": float(total_amount),
            "late_payment_probability": round(prob_late, 4),
            "is_suspected_late_payment": is_suspected_late,
            "risk_advisory": "High likelihood of settlement delay" if is_suspected_late else "Normal settlement expected within terms",
            "vendor_prior_late_rate": round(prior_rate, 4)
        }
