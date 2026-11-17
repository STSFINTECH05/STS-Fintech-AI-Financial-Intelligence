"""
STS FINTECH - PERSONAL FINANCE CLASSIFIER ROUTES
Milestone 2 - Step 7.4C-3D (Experimental Microservice)

Defines the isolated, HMAC-protected internal endpoint:
POST /internal/v1/personal/classify

Directly reuses the canonical verify_internal_hmac dependency from ml.service.main
to guarantee zero divergence, identical header enforcement, and identical replay tracking.
"""

import time
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, Request, HTTPException, status

from ml.service.personal_schemas import PersonalClassifyRequest, PersonalClassifyResponse
from ml.service.prediction.personal_classifier import PersonalFinanceClassifierService
from ml.service.main import verify_internal_hmac

logger = logging.getLogger("sts_ml_service.personal_routes")

router = APIRouter(prefix="/internal/v1/personal", tags=["Personal Finance Classifier (Experimental)"])

# Lazy singleton accessor
_personal_service: Optional[PersonalFinanceClassifierService] = None


def get_personal_classifier_service() -> PersonalFinanceClassifierService:
    """Lazy accessor ensuring model bundle is deserialized only upon demand."""
    global _personal_service
    if _personal_service is None:
        _personal_service = PersonalFinanceClassifierService.get_instance()
    return _personal_service


@router.post(
    "/classify",
    response_model=PersonalClassifyResponse,
    dependencies=[Depends(verify_internal_hmac)]
)
async def classify_personal_transaction(request_body: PersonalClassifyRequest):
    """
    Experimental internal inference endpoint predicting category suggestions
    for personal and small-business income and expense transactions.
    Enforces pre-vectorization safety guardrails and strict human-confirmation requirements.
    Reuses canonical verify_internal_hmac dependency from ml.service.main.
    """
    start_time = time.perf_counter()
    service = get_personal_classifier_service()

    result = service.classify_transaction(
        transaction_direction=request_body.transaction_direction,
        description=request_body.description,
        amount_inr=request_body.amount_inr,
        transaction_date=request_body.transaction_date
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Minimal non-PII operational log: direction, status, category, latency
    logger.info(
        "Personal classification: direction=%s status=%s category=%s latency_ms=%.2f",
        request_body.transaction_direction,
        result.get("status"),
        result.get("category"),
        elapsed_ms
    )

    return result
