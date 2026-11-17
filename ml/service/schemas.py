"""
STS FINTECH AI FINANCIAL INTELLIGENCE
API Schema Contracts & Data Transfer Objects (Milestone 2 - Step 7.4B-1)

Defines strict request/response schemas for health and model registry endpoints.
Enforces explicit data provenance, version metadata, and safety indicators.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

# Graceful Pydantic v2 / v1 support with dataclass fallback if pydantic is not yet installed
try:
    from pydantic import BaseModel, Field

    class LiveHealthResponse(BaseModel):
        service_status: str = Field(default="ALIVE", description="Process liveness status")
        timestamp: str = Field(description="ISO 8601 server timestamp")

    class ModelArtifactStatus(BaseModel):
        model_id: str
        model_name: str
        model_version: str
        status: str = Field(description="LOADED, DEFERRED, or EXCLUDED")
        sha256_verified: bool
        data_provenance: str = Field(default="SYNTHETIC_V2_DATASET")

    class InternalHealthResponse(BaseModel):
        service_status: str
        model_version: str
        models_loaded_count: int
        artifact_verification_status: str
        data_provenance: str = Field(default="SYNTHETIC_V2_DATASET")
        timestamp: str

    class ModelRegistryStatusResponse(BaseModel):
        service_status: str
        model_version: str
        data_provenance: str = Field(default="SYNTHETIC_V2_DATASET")
        timestamp: str
        models: Dict[str, ModelArtifactStatus]

    class ErrorResponse(BaseModel):
        success: bool = Field(default=False)
        error_code: str
        message: str
        timestamp: str

    class ClassificationRequest(BaseModel):
        amount: float = Field(gt=0.0, description="Transaction expenditure amount (positive, finite)")
        description: Optional[str] = Field(default="", max_length=255, description="Merchant transaction description narrative")
        transaction_date: str = Field(description="Transaction date formatted as YYYY-MM-DD")
        department_id: str = Field(min_length=1, description="Mandatory organizational department identifier")
        vendor_category: Optional[str] = Field(default="General Supplier", description="Vendor business sector or category")
        division: Optional[str] = Field(default="General Operations", description="Company operating division")
        payment_method: Optional[str] = Field(default="OTHER", description="Payment instrument")
        transaction_type: Optional[str] = Field(default="PURCHASE", description="Operational transaction type")
        vendor_rating: Optional[float] = Field(default=3.5, ge=1.0, le=5.0, description="Vendor performance rating")
        payment_terms_days: Optional[int] = Field(default=30, ge=0, description="Payment terms in days")
        annual_budget_baseline: Optional[float] = Field(default=10000000.0, ge=0.0, description="Department annual budget baseline")
        is_preferred_vendor: Optional[bool] = Field(default=False, description="Preferred vendor status flag")
        employment_type: Optional[str] = Field(default="FULL_TIME", description="Employee arrangement")

    class CategoryAlternative(BaseModel):
        category: str
        raw_decision_score: float

    class InferenceProvenance(BaseModel):
        model_version: str = Field(default="2.0.0")
        model_name: str = Field(default="LogisticRegression")
        artifact_sha256: str
        data_provenance: str = Field(default="SYNTHETIC_V2_DATASET")
        inference_latency_ms: float

    class ClassificationResponse(BaseModel):
        status: str = Field(default="SUCCESS")
        suggested_category: str
        raw_decision_score: float
        score_interpretation: str = Field(default="RAW_UNCALIBRATED_MODEL_SCORE")
        requires_manual_confirmation: bool = Field(default=True)
        alternatives: List[CategoryAlternative] = Field(default_factory=list)
        inference_status: str = Field(default="EVALUATED")
        provenance: InferenceProvenance

    class AnomalyScoreRequest(BaseModel):
        amount: float = Field(gt=0.0, description="Transaction expenditure amount (positive, finite)")
        transaction_date: str = Field(description="Transaction date formatted as YYYY-MM-DD")
        category: str = Field(min_length=1, description="Assigned or suggested expense category")
        department_id: str = Field(min_length=1, description="Mandatory organizational department identifier")
        category_median: float = Field(gt=0.0, description="Baseline category spending median (mandatory context)")
        department_median: float = Field(gt=0.0, description="Baseline department spending median (mandatory context)")
        spending_approval_limit: float = Field(gt=0.0, description="Employee spending approval limit (mandatory context)")
        annual_budget_baseline: float = Field(ge=0.0, description="Department annual budget baseline (mandatory context)")
        vendor_rating: float = Field(ge=1.0, le=5.0, description="Vendor performance rating (mandatory context)")
        payment_terms_days: Optional[int] = Field(default=30, ge=0, description="Payment terms in days")
        vendor_category: Optional[str] = Field(default="General Supplier", description="Vendor sector/category")
        division: Optional[str] = Field(default="General Operations", description="Company operating division")
        payment_method: Optional[str] = Field(default="OTHER", description="Payment instrument")
        transaction_type: Optional[str] = Field(default="PURCHASE", description="Operational transaction type")
        is_preferred_vendor: Optional[bool] = Field(default=False, description="Preferred vendor status flag")
        employment_type: Optional[str] = Field(default="FULL_TIME", description="Employee arrangement")

    class AnomalyScoreResponse(BaseModel):
        status: str = Field(default="SUCCESS")
        inference_status: str = Field(default="EVALUATED")
        anomaly_indicator: int = Field(description="Binary flag: 1 if anomaly score >= threshold, else 0")
        anomaly_score: Optional[float] = Field(description="Normalized anomaly score in [0.0, 1.0]")
        calibrated_threshold: float = Field(default=0.51, description="Calibrated decision threshold")
        advisory_classification: str = Field(description="Low Variance, Moderate Discrepancy, or High Statistical Anomaly – Requires Verification")
        review_recommended: bool = Field(default=False, description="Flag indicating whether review is advised")
        mandatory_human_review: bool = Field(default=True, description="Strict requirement for human verification")
        disclaimer: str = Field(description="Standard statistical disclaimer")
        provenance: InferenceProvenance

    class DailySeriesItem(BaseModel):
        date: str = Field(description="Observation date (YYYY-MM-DD)")
        daily_expenditure: float = Field(ge=0.0, description="Daily aggregated expense amount")

    class ForecastRequest(BaseModel):
        as_of_date: str = Field(description="Cutoff forecast origin date (YYYY-MM-DD)")
        horizon_days: int = Field(default=30, ge=1, le=90, description="Forecast horizon in days (1 to 90)")
        currency: str = Field(default="INR", description="Currency code (strictly 'INR')")
        historical_daily_series: List[DailySeriesItem] = Field(description="Continuous daily expenditure series (>= 30 days)")

    class HeuristicPlanningRange(BaseModel):
        lower: float = Field(ge=0.0, description="Heuristic lower planning bound")
        upper: float = Field(ge=0.0, description="Heuristic upper planning bound")

    class ProjectionItem(BaseModel):
        step: int = Field(ge=1, description="Projection horizon day step (1 to horizon)")
        date: str = Field(description="Projected calendar date (YYYY-MM-DD)")
        predicted_expenditure: float = Field(ge=0.0, description="Projected expenditure amount")
        heuristic_planning_range: HeuristicPlanningRange

    class ForecastResponse(BaseModel):
        status: str = Field(default="SUCCESS")
        inference_status: str = Field(default="EVALUATED")
        forecast_origin: str = Field(description="Cutoff origin date (YYYY-MM-DD)")
        horizon_days: int = Field(ge=1, le=90, description="Number of projected days")
        currency: str = Field(default="INR")
        total_projected_expenditure: Optional[float] = Field(description="Sum of projected daily expenditures")
        average_daily_expenditure: Optional[float] = Field(description="Average daily projected expenditure")
        projections: List[ProjectionItem] = Field(default_factory=list)
        planning_range_type: str = Field(default="HEURISTIC_PLANNING_RANGE")
        provenance: InferenceProvenance

    class BudgetPredictRequest(BaseModel):
        department_id: str = Field(min_length=1, description="Mandatory organizational department identifier (e.g. '1' to '50')")
        budget_year: int = Field(ge=2000, le=2100, description="Four-digit budget calendar year (e.g. 2026)")
        budget_month: int = Field(ge=1, le=12, description="Budget calendar month (1 to 12)")
        as_of_day: int = Field(ge=1, le=31, description="Supported snapshot horizon day (strictly 7, 14, or 21)")
        allocated_amount: float = Field(gt=0.0, description="Approved departmental monthly budget allocation (positive, finite)")
        month_to_date_actual: float = Field(ge=0.0, description="Actual expenditure incurred month-to-date (non-negative, finite)")
        transaction_count_mtd: Optional[int] = Field(default=None, ge=0, description="Total transactions posted month-to-date")
        annual_budget_baseline: Optional[float] = Field(default=None, gt=0.0, description="Annual baseline allocation for department")

    class BudgetPredictResponse(BaseModel):
        status: str = Field(default="SUCCESS")
        inference_status: str = Field(default="EVALUATED")
        department_id: str = Field(description="Department identifier")
        budget_year: int = Field(description="Budget calendar year")
        budget_month: int = Field(description="Budget calendar month")
        as_of_day: int = Field(description="Snapshot horizon day")
        allocated_amount: float = Field(description="Approved departmental monthly allocation")
        month_to_date_actual: float = Field(description="Actual expenditure incurred month-to-date")
        predicted_final_actual: Optional[float] = Field(description="Estimated final monthly expenditure")
        predicted_utilization_pct: Optional[float] = Field(description="Predicted final budget utilization percentage")
        estimated_remaining_budget: Optional[float] = Field(description="Estimated remaining unspent budget allocation")
        advisory_risk_tier: str = Field(description="Deterministic risk tier: Within Budget, Approaching Budget Limit, or Potential Budget Overrun – Requires Verification")
        advisory_notice: str = Field(default="Informational spending pace indicator only. Human financial review required.")
        provenance: InferenceProvenance

except ImportError:
    # Dataclass fallback when pydantic package is not yet installed in .venv
    @dataclass
    class LiveHealthResponse:
        service_status: str
        timestamp: str

        def dict(self):
            return asdict(self)

    @dataclass
    class ModelArtifactStatus:
        model_id: str
        model_name: str
        model_version: str
        status: str
        sha256_verified: bool
        data_provenance: str = "SYNTHETIC_V2_DATASET"

        def dict(self):
            return asdict(self)

    @dataclass
    class InternalHealthResponse:
        service_status: str
        model_version: str
        models_loaded_count: int
        artifact_verification_status: str
        timestamp: str
        data_provenance: str = "SYNTHETIC_V2_DATASET"

        def dict(self):
            return asdict(self)

    @dataclass
    class ModelRegistryStatusResponse:
        service_status: str
        model_version: str
        timestamp: str
        models: Dict[str, Any]
        data_provenance: str = "SYNTHETIC_V2_DATASET"

        def dict(self):
            return asdict(self)

    @dataclass
    class ErrorResponse:
        error_code: str
        message: str
        timestamp: str
        success: bool = False

        def dict(self):
            return asdict(self)

    @dataclass
    class ClassificationRequest:
        amount: float
        transaction_date: str
        department_id: str
        description: Optional[str] = ""
        vendor_category: Optional[str] = "General Supplier"
        division: Optional[str] = "General Operations"
        payment_method: Optional[str] = "OTHER"
        transaction_type: Optional[str] = "PURCHASE"
        vendor_rating: Optional[float] = 3.5
        payment_terms_days: Optional[int] = 30
        annual_budget_baseline: Optional[float] = 10000000.0
        is_preferred_vendor: Optional[bool] = False
        employment_type: Optional[str] = "FULL_TIME"

        def dict(self):
            return asdict(self)

    @dataclass
    class CategoryAlternative:
        category: str
        raw_decision_score: float

        def dict(self):
            return asdict(self)

    @dataclass
    class InferenceProvenance:
        artifact_sha256: str
        inference_latency_ms: float
        model_version: str = "2.0.0"
        model_name: str = "LogisticRegression"
        data_provenance: str = "SYNTHETIC_V2_DATASET"

        def dict(self):
            return asdict(self)

    @dataclass
    class ClassificationResponse:
        suggested_category: str
        raw_decision_score: float
        provenance: InferenceProvenance
        status: str = "SUCCESS"
        score_interpretation: str = "RAW_UNCALIBRATED_MODEL_SCORE"
        requires_manual_confirmation: bool = True
        alternatives: List[Any] = None
        inference_status: str = "EVALUATED"

        def dict(self):
            return asdict(self)

    @dataclass
    class AnomalyScoreRequest:
        amount: float
        transaction_date: str
        category: str
        department_id: str
        category_median: float
        department_median: float
        spending_approval_limit: float
        annual_budget_baseline: float
        vendor_rating: float
        payment_terms_days: Optional[int] = 30
        vendor_category: Optional[str] = "General Supplier"
        division: Optional[str] = "General Operations"
        payment_method: Optional[str] = "OTHER"
        transaction_type: Optional[str] = "PURCHASE"
        is_preferred_vendor: Optional[bool] = False
        employment_type: Optional[str] = "FULL_TIME"

        def dict(self):
            return asdict(self)

    @dataclass
    class AnomalyScoreResponse:
        status: str
        inference_status: str
        anomaly_indicator: int
        anomaly_score: Optional[float]
        calibrated_threshold: float
        advisory_classification: str
        review_recommended: bool
        mandatory_human_review: bool
        disclaimer: str
        provenance: InferenceProvenance

        def dict(self):
            return asdict(self)

    @dataclass
    class DailySeriesItem:
        date: str
        daily_expenditure: float

        def dict(self):
            return asdict(self)

    @dataclass
    class ForecastRequest:
        as_of_date: str
        historical_daily_series: List[DailySeriesItem]
        horizon_days: int = 30
        currency: str = "INR"

        def dict(self):
            return asdict(self)

    @dataclass
    class HeuristicPlanningRange:
        lower: float
        upper: float

        def dict(self):
            return asdict(self)

    @dataclass
    class ProjectionItem:
        step: int
        date: str
        predicted_expenditure: float
        heuristic_planning_range: HeuristicPlanningRange

        def dict(self):
            return asdict(self)

    @dataclass
    class ForecastResponse:
        status: str
        inference_status: str
        forecast_origin: str
        horizon_days: int
        currency: str
        total_projected_expenditure: Optional[float]
        average_daily_expenditure: Optional[float]
        provenance: InferenceProvenance
        projections: List[ProjectionItem] = None
        planning_range_type: str = "HEURISTIC_PLANNING_RANGE"

        def dict(self):
            return asdict(self)

    @dataclass
    class BudgetPredictRequest:
        department_id: str
        budget_year: int
        budget_month: int
        as_of_day: int
        allocated_amount: float
        month_to_date_actual: float
        transaction_count_mtd: Optional[int] = None
        annual_budget_baseline: Optional[float] = None

        def dict(self):
            return asdict(self)

    @dataclass
    class BudgetPredictResponse:
        status: str
        inference_status: str
        department_id: str
        budget_year: int
        budget_month: int
        as_of_day: int
        allocated_amount: float
        month_to_date_actual: float
        predicted_final_actual: Optional[float]
        predicted_utilization_pct: Optional[float]
        estimated_remaining_budget: Optional[float]
        advisory_risk_tier: str
        advisory_notice: str
        provenance: InferenceProvenance

        def dict(self):
            return asdict(self)
