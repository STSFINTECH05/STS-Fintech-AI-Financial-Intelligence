"""
STS FINTECH - PERSONAL FINANCE CLASSIFIER SCHEMAS
Milestone 2 - Step 7.4C-3D (Experimental Microservice)

Defines strict Pydantic schemas for the experimental personal finance
category classification endpoint (/internal/v1/personal/classify).
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator
import datetime
import re

# ISO 8601 YYYY-MM-DD validator regex
DATE_REGEX = re.compile(r'^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$')


class PersonalClassifyRequest(BaseModel):
    """
    Input schema for personal finance transaction classification.
    Text-only inference pipeline: description is the predictive signal.
    amount_inr and transaction_date are validated for safety and OOD boundaries.
    """
    transaction_direction: Literal["income", "expense"] = Field(
        ...,
        description="Direction of transaction ('income' or 'expense')."
    )
    description: str = Field(
        ...,
        max_length=500,
        description="Transaction narrative / bank narration (max 500 characters)."
    )
    amount_inr: float = Field(
        ...,
        ge=0.0,
        description="Transaction amount in INR. Must be non-negative numeric value."
    )
    transaction_date: str = Field(
        ...,
        description="Transaction date in strict ISO-8601 format (YYYY-MM-DD). Never substituted."
    )

    @field_validator("transaction_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        if not isinstance(v, str) or not DATE_REGEX.match(v.strip()):
            raise ValueError("Invalid transaction_date format. Must be strict ISO-8601 YYYY-MM-DD.")
        # Verify calendar validity
        parts = v.strip().split("-")
        try:
            datetime.date(int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError as e:
            raise ValueError(f"Invalid calendar date: {v}. {str(e)}")
        return v.strip()

    @field_validator("description")
    @classmethod
    def validate_description_length(cls, v: str) -> str:
        if v is None:
            return ""
        return v.strip()


class PersonalClassifyResponse(BaseModel):
    """
    Output schema for personal finance category suggestion.
    Strictly marked requires_manual_confirmation: true.
    ranking_score is explicitly defined as INTERNAL_RANKING_ONLY.
    """
    status: Literal["SUCCESS", "ABSTAIN", "NOT_EVALUATED"] = Field(
        ...,
        description="Execution status of classification attempt."
    )
    category: Optional[str] = Field(
        None,
        description="Predicted category suggestion if status is SUCCESS, else None."
    )
    ranking_score: Optional[float] = Field(
        None,
        description="Internal model ranking score (not a calibrated probability). None if abstained."
    )
    score_semantics: str = Field(
        "INTERNAL_RANKING_ONLY",
        description="Semantic disclaimer defining the score as an uncalibrated ranking signal."
    )
    requires_manual_confirmation: bool = Field(
        True,
        description="Invariant: all suggestions require human user confirmation."
    )
    abstention_reason: Optional[str] = Field(
        None,
        description="Reason for abstention (e.g. UNSUPPORTED_INCOME_SALARY, AMBIGUOUS, etc.)."
    )
    model_version: str = Field(
        "personal_finance_bundle_v1_candidate",
        description="Model artifact identifier."
    )
