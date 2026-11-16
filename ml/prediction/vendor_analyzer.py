"""
STS Fintech – Milestone 2 – Step 7.2D: Vendor Risk & Spending Analysis Inference Module
Provides enterprise vendor risk screening, friction indicator breakdowns,
and accounts payable review recommendations.
"""

import os
import joblib
import pandas as pd
import numpy as np


MODEL_PATH = "ml/models/vendor_analyzer.joblib"


class VendorAnalyzer:
    """
    Transparent rule-based vendor risk screening and expenditure analytics engine.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Vendor analyzer bundle not found at: {model_path}")

        bundle = joblib.load(model_path)
        self.algorithm = bundle.get("algorithm", "EnterpriseMultiFactorRuleScreeningEngine")
        self.scoring_weights = bundle.get("scoring_weights", {})
        self.scoring_thresholds = bundle.get("scoring_thresholds", {"low_threshold": 30.0, "medium_threshold": 60.0})
        self.profiles = bundle.get("profiles", {})
        self.audit = bundle.get("audit", {})
        self.tier_distribution = bundle.get("tier_distribution", {})

    def analyze_vendor(self, vendor_id: str) -> dict:
        """
        Retrieves comprehensive expenditure, billing indicators, and risk tier for a vendor.

        Args:
            vendor_id: Vendor identifier (e.g., 'VEN00001').

        Returns:
            Dictionary containing financial profile, risk score, factor breakdown, and review recommendation.
        """
        if not vendor_id or not isinstance(vendor_id, str):
            raise ValueError(f"vendor_id must be a non-empty string, got: {vendor_id}")

        vid = vendor_id.strip().upper()
        if vid not in self.profiles:
            raise KeyError(f"Vendor ID '{vid}' not found in registered enterprise vendor registry.")

        p = self.profiles[vid]

        # Determine recommended review action
        tier = p["risk_tier"]
        if tier == "LOW":
            action = "Standard automated three-way matching and scheduled disbursement."
        elif tier == "MEDIUM":
            action = "Routine automated validation with periodic accounts payable sampling."
        else:
            action = "Mandatory manual verification of pending invoices and vendor documentation prior to payment authorization."

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
                "accounting_note": "Invoice, bill, and transaction figures represent distinct recorded ledgers and must not be summed together."
            },
            "friction_indicators": {
                "rejected_invoices_count": p["rejected_invoices"],
                "duplicate_invoices_count": p["duplicate_invoices"],
                "overdue_invoices_count": p["overdue_invoices"],
                "overdue_bills_count": p["overdue_bills"],
                "transaction_anomalies_count": p["anomaly_tx_count"]
            },
            "risk_evaluation": {
                "risk_score": p["risk_score"],
                "risk_tier": p["risk_tier"],
                "factor_breakdown": p["score_breakdown"],
                "scoring_weights": self.scoring_weights
            },
            "recommended_action": action,
            "compliance_disclaimer": "Statistical risk indicator for screening prioritization only. Does not constitute a determination of fraud, default, or contractual breach."
        }

    def get_high_risk_vendors(self) -> list:
        """Returns all registered vendors classified as HIGH – REQUIRES VERIFICATION."""
        high_risk = []
        for vid, p in self.profiles.items():
            if p["risk_tier"] == "HIGH – REQUIRES VERIFICATION":
                high_risk.append({
                    "vendor_id": vid,
                    "vendor_name": p["vendor_name"],
                    "vendor_category": p["vendor_category"],
                    "risk_score": p["risk_score"],
                    "duplicate_invoices": p["duplicate_invoices"],
                    "rejected_invoices": p["rejected_invoices"],
                    "anomaly_tx_count": p["anomaly_tx_count"]
                })
        return sorted(high_risk, key=lambda x: x["risk_score"], reverse=True)

    def screen_all_vendors(self) -> pd.DataFrame:
        """Returns summary dataframe of all 2,000 vendor profiles."""
        rows = []
        for vid, p in self.profiles.items():
            rows.append({
                "vendor_id": vid,
                "vendor_name": p["vendor_name"],
                "vendor_category": p["vendor_category"],
                "vendor_rating": p["vendor_rating"],
                "invoice_count": p["invoice_count"],
                "total_invoice_amt": p["total_invoice_amt"],
                "duplicate_invoices": p["duplicate_invoices"],
                "rejected_invoices": p["rejected_invoices"],
                "risk_score": p["risk_score"],
                "risk_tier": p["risk_tier"]
            })
        return pd.DataFrame(rows)
