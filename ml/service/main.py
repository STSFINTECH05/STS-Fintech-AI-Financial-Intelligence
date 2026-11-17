"""
STS FINTECH AI FINANCIAL INTELLIGENCE
Internal FastAPI Microservice Application (Milestone 2 - Step 7.4B-1)

Private internal ML service providing verified V2 model inference and health diagnostics.
Enforces:
- SHA-256 pre-execution artifact verification at startup.
- Cryptographic HMAC-SHA256 request authentication on all internal routes.
- Strict synthetic data provenance indicators.
- Zero local disk modifications or database connections.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from ml.service.config import config
from ml.service.model_loader import ModelRegistry, ModelIntegrityError
from ml.service.security.hmac_auth import verify_hmac_request
from ml.service.schemas import (
    LiveHealthResponse,
    InternalHealthResponse,
    ModelRegistryStatusResponse,
    ErrorResponse,
    ClassificationRequest,
    ClassificationResponse,
    AnomalyScoreRequest,
    AnomalyScoreResponse,
    ForecastRequest,
    ForecastResponse,
    BudgetPredictRequest,
    BudgetPredictResponse
)
from ml.service.prediction.expense_classifier import ExpenseClassifierV2, InputValidationError
from ml.service.prediction.anomaly_detector import AnomalyDetectorV2, AnomalyInputValidationError
from ml.service.prediction.expense_forecaster import ExpenseForecasterService, ForecastInputValidationError
from ml.service.prediction.budget_predictor import BudgetPredictorV2Service, BudgetInputValidationError

# Configure structured logging (ensuring secrets and PII are never logged)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("sts_ml_service")

# Global in-memory model registry
registry = ModelRegistry(models_dir=config.models_dir)

# Check FastAPI availability
try:
    from fastapi import FastAPI, Request, HTTPException, Depends, status
    from fastapi.responses import JSONResponse
    from contextlib import asynccontextmanager
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


def get_model_registry() -> ModelRegistry:
    """Dependency provider for in-memory model registry."""
    return registry


async def verify_internal_hmac(request: "Request") -> None:
    """
    FastAPI dependency enforcing HMAC-SHA256 authentication on internal endpoints.
    Validates headers: X-STS-Timestamp, X-STS-Nonce, X-STS-Signature.
    """
    timestamp_str = request.headers.get("X-STS-Timestamp")
    nonce = request.headers.get("X-STS-Nonce")
    signature = request.headers.get("X-STS-Signature")

    if not timestamp_str or not nonce or not signature:
        logger.warning("Unauthenticated internal request: Missing HMAC security headers.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "success": False,
                "error_code": "MISSING_SECURITY_HEADERS",
                "message": "Required headers missing: X-STS-Timestamp, X-STS-Nonce, X-STS-Signature",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )

    body = await request.body()
    active_key = config.get_active_key()
    previous_key = config.get_previous_key()

    is_valid, error_code = verify_hmac_request(
        method=request.method,
        path=request.url.path,
        timestamp_str=timestamp_str,
        nonce=nonce,
        signature=signature,
        body=body,
        active_key=active_key,
        previous_key=previous_key,
        replay_window_seconds=config.replay_window_seconds
    )

    if not is_valid:
        logger.warning("HMAC authentication rejected for %s: %s", request.url.path, error_code)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "success": False,
                "error_code": error_code or "AUTH_FAILED",
                "message": "HMAC-SHA256 signature verification failed or expired.",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )


def create_app() -> Any:
    """
    Factory function creating and configuring the FastAPI microservice.
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError(
            "FastAPI is not installed in the active Python environment. "
            "Please review dependencies and install fastapi/uvicorn."
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: verify SHA-256 and load approved models
        logger.info("Initializing STS ML Microservice v2.0.0...")
        logger.info("Verifying model artifact checksums in %s...", config.models_dir)
        try:
            registry.initialize_registry()
            logger.info(
                "Model Registry initialized successfully. Loaded %d models.",
                len(registry.loaded_models)
            )
            global classifier_service, anomaly_service, forecaster_service, budget_service
            classifier_service = ExpenseClassifierV2(registry)
            anomaly_service = AnomalyDetectorV2(registry)
            forecaster_service = ExpenseForecasterService(registry)
            budget_service = BudgetPredictorV2Service(registry)
        except Exception as e:
            logger.critical("Fatal error during model registry startup: %s", str(e))
            raise e
        yield
        # Shutdown
        logger.info("Shutting down STS ML Microservice.")

    app = FastAPI(
        title="STS Fintech AI Financial Intelligence Microservice",
        version="2.0.0",
        description="Private internal service for V2 financial ML inference.",
        lifespan=lifespan,
        docs_url=None,  # Disabled on internal production microservice for security
        redoc_url=None
    )

    # 1. Public Liveness Probe
    @app.get("/health/live")
    async def health_live():
        """Public liveness probe to verify process availability."""
        now_iso = datetime.utcnow().isoformat() + "Z"
        return {
            "service_status": "ALIVE",
            "timestamp": now_iso
        }

    # 2. Protected Internal Health & Readiness Probe
    @app.get("/internal/v2/health", dependencies=[Depends(verify_internal_hmac)])
    async def internal_health():
        """Protected readiness probe verifying model loading and system health."""
        summary = registry.get_status_summary()
        now_iso = datetime.utcnow().isoformat() + "Z"
        return {
            "service_status": summary["service_status"],
            "model_version": summary["model_version"],
            "models_loaded_count": summary["models_loaded_count"],
            "artifact_verification_status": summary["artifact_verification_status"],
            "data_provenance": summary["data_provenance"],
            "timestamp": now_iso
        }

    # 3. Protected Detailed Model Registry Status
    @app.get("/internal/v2/models/status", dependencies=[Depends(verify_internal_hmac)])
    async def internal_models_status():
        """Protected registry endpoint detailing approved, deferred, and excluded models."""
        summary = registry.get_status_summary()
        now_iso = datetime.utcnow().isoformat() + "Z"
        return {
            "service_status": summary["service_status"],
            "model_version": summary["model_version"],
            "data_provenance": summary["data_provenance"],
            "timestamp": now_iso,
            "models": summary["models"]
        }

    # 4. Protected Expense Classification Endpoint (Model A)
    @app.post(
        "/internal/v2/classify",
        response_model=ClassificationResponse,
        dependencies=[Depends(verify_internal_hmac)]
    )
    async def classify_expense_endpoint(request_body: ClassificationRequest):
        """Protected endpoint predicting expense category from transaction features."""
        global classifier_service
        if classifier_service is None:
            classifier_service = ExpenseClassifierV2(registry)

        payload = request_body.model_dump() if hasattr(request_body, "model_dump") else request_body.dict()
        try:
            return classifier_service.classify_expense(payload)
        except InputValidationError as e:
            logger.warning("Classification input validation failure: %s", str(e))
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "success": False,
                    "error_code": "INVALID_INPUT_FEATURE",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
            )
        except Exception as e:
            logger.error("Internal error executing classification inference: %s", str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "success": False,
                    "error_code": "INFERENCE_ERROR",
                    "message": "Internal error executing expense classification.",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
            )

    # 5. Protected Anomaly Detection Endpoint (Model B)
    @app.post(
        "/internal/v2/anomaly/score",
        response_model=AnomalyScoreResponse,
        dependencies=[Depends(verify_internal_hmac)]
    )
    async def score_anomaly_endpoint(request_body: AnomalyScoreRequest):
        """Protected endpoint calculating statistical anomaly score and advisory risk tier."""
        global anomaly_service
        if anomaly_service is None:
            anomaly_service = AnomalyDetectorV2(registry)

        payload = request_body.model_dump() if hasattr(request_body, "model_dump") else request_body.dict()
        try:
            return anomaly_service.score_transaction(payload)
        except AnomalyInputValidationError as e:
            logger.warning("Anomaly input validation failure: %s", str(e))
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "success": False,
                    "error_code": "INVALID_INPUT_FEATURE",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
            )
        except Exception as e:
            logger.error("Internal error executing anomaly scoring: %s", str(e))
            return {
                "status": "ERROR",
                "inference_status": "NOT_EVALUATED",
                "anomaly_indicator": 0,
                "anomaly_score": None,
                "calibrated_threshold": 0.51,
                "advisory_classification": "NOT_EVALUATED",
                "review_recommended": True,
                "mandatory_human_review": True,
                "disclaimer": (
                    "Unsupervised statistical screening indicator based on synthetic training data. "
                    "This is not a determination of fraud, misconduct, or financial policy violation. "
                    "Human review is required."
                ),
                "provenance": {
                    "model_version": "2.0.0",
                    "model_name": "IsolationForestV2AnomalyDetector",
                    "artifact_sha256": "e7094d751b337862bc6892c551f215dea0957a1b7e1926a851f7425837c22f29",
                    "data_provenance": "SYNTHETIC_V2_DATASET",
                    "inference_latency_ms": 0.0
                }
            }

    # 6. Protected Expense Forecasting Endpoint (Model C)
    @app.post(
        "/internal/v2/forecast",
        response_model=ForecastResponse,
        dependencies=[Depends(verify_internal_hmac)]
    )
    async def forecast_expense_endpoint(request_body: ForecastRequest):
        """Protected endpoint generating recursive multi-day expenditure forecasts."""
        global forecaster_service
        if forecaster_service is None:
            forecaster_service = ExpenseForecasterService(registry)

        payload = request_body.model_dump() if hasattr(request_body, "model_dump") else request_body.dict()
        try:
            return forecaster_service.forecast_expense(payload)
        except ForecastInputValidationError as e:
            logger.warning("Forecast input validation failure: %s", str(e))
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "success": False,
                    "error_code": "INVALID_INPUT_FEATURE",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
            )
        except Exception as e:
            logger.error("Internal error executing expense forecast: %s", str(e))
            return {
                "status": "ERROR",
                "inference_status": "NOT_EVALUATED",
                "forecast_origin": str(payload.get("as_of_date", "")),
                "horizon_days": int(payload.get("horizon_days", 30)),
                "currency": str(payload.get("currency", "INR")),
                "total_projected_expenditure": None,
                "average_daily_expenditure": None,
                "projections": [],
                "planning_range_type": "HEURISTIC_PLANNING_RANGE",
                "provenance": {
                    "model_name": "RandomForestRegressor",
                    "model_version": "2.0.0",
                    "artifact_sha256": "ea71b872864f3ab8dba53ab96422e24c6d21b730c9fee0191b9fded950de6b22",
                    "model_training_data_provenance": "SYNTHETIC_V2_DATASET",
                    "inference_latency_ms": 0.0
                }
            }

    # 7. Protected Budget Utilization Prediction Endpoint (Model D)
    @app.post(
        "/internal/v2/budget/predict",
        response_model=BudgetPredictResponse,
        dependencies=[Depends(verify_internal_hmac)]
    )
    async def predict_budget_endpoint(request_body: BudgetPredictRequest):
        """Protected endpoint predicting departmental budget utilization and remaining allocation."""
        global budget_service
        if budget_service is None:
            budget_service = BudgetPredictorV2Service(registry)

        payload = request_body.model_dump() if hasattr(request_body, "model_dump") else request_body.dict()
        try:
            return budget_service.predict_budget(payload)
        except BudgetInputValidationError as e:
            logger.warning("Budget input validation failure: %s", str(e))
            error_code = "UNSUPPORTED_SNAPSHOT_HORIZON" if "Unsupported snapshot horizon" in str(e) else "INVALID_INPUT_FEATURE"
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "success": False,
                    "error_code": error_code,
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
            )
        except Exception as e:
            logger.error("Internal error executing budget prediction: %s", str(e))
            return {
                "status": "ERROR",
                "inference_status": "NOT_EVALUATED",
                "department_id": str(payload.get("department_id", "")),
                "budget_year": int(payload.get("budget_year", 2026)) if payload.get("budget_year") else 2026,
                "budget_month": int(payload.get("budget_month", 1)) if payload.get("budget_month") else 1,
                "as_of_day": int(payload.get("as_of_day", 14)) if payload.get("as_of_day") else 14,
                "allocated_amount": float(payload.get("allocated_amount", 0.0)) if payload.get("allocated_amount") else 0.0,
                "month_to_date_actual": float(payload.get("month_to_date_actual", 0.0)) if payload.get("month_to_date_actual") else 0.0,
                "predicted_final_actual": None,
                "predicted_utilization_pct": None,
                "estimated_remaining_budget": None,
                "advisory_risk_tier": "NOT_EVALUATED",
                "advisory_notice": "Informational spending pace indicator only. Human financial review required.",
                "provenance": {
                    "model_name": "RandomForestRegressor",
                    "model_version": "2.0.0",
                    "artifact_sha256": "3a56c7c086b01814fdce1367e967292122c9ad2507bae1023c9a26e1dc8e3199",
                    "data_provenance": "SYNTHETIC_V2_DATASET",
                    "inference_latency_ms": 0.0
                }
            }

    # 8. Protected Personal Finance Classifier Endpoint (Milestone 2 - Step 7.4C-3D Experimental)
    from ml.service.routes.personal_routes import router as personal_router
    app.include_router(personal_router)

    return app


# Global service references
classifier_service: Optional[ExpenseClassifierV2] = None
anomaly_service: Optional[AnomalyDetectorV2] = None
forecaster_service: Optional[ExpenseForecasterService] = None
budget_service: Optional[BudgetPredictorV2Service] = None

# Module-level app instance if FastAPI is available
app = None
if FASTAPI_AVAILABLE:
    app = create_app()
