"""
STS FINTECH AI FINANCIAL INTELLIGENCE
Service Health and Schema Contract Tests (Milestone 2 - Step 7.4B-1)

Verifies:
1. Schema contract compliance and data provenance indicators.
2. Health response structures and status codes.
3. Model registry readiness state and status metadata.
4. FastAPI application creation and dependency verification.
"""

from datetime import datetime
import pytest

from ml.service.schemas import (
    LiveHealthResponse,
    InternalHealthResponse,
    ModelRegistryStatusResponse,
    ModelArtifactStatus
)
from ml.service.config import DEFAULT_MODELS_DIR
from ml.service.model_loader import ModelRegistry
from ml.service.main import FASTAPI_AVAILABLE, create_app


class TestHealthAndSchemas:
    """Test suite for health contracts, schemas, and service status."""

    def test_live_health_schema_contract(self):
        """Verifies LiveHealthResponse structure."""
        now_iso = datetime.utcnow().isoformat() + "Z"
        res = LiveHealthResponse(
            service_status="ALIVE",
            timestamp=now_iso
        )
        data = res.model_dump() if hasattr(res, "model_dump") else (res.dict() if hasattr(res, "dict") else res.__dict__)
        assert data["service_status"] == "ALIVE"
        assert "timestamp" in data

    def test_internal_health_schema_data_provenance(self):
        """Verifies InternalHealthResponse enforces SYNTHETIC_V2_DATASET."""
        now_iso = datetime.utcnow().isoformat() + "Z"
        res = InternalHealthResponse(
            service_status="READY",
            model_version="2.0.0",
            models_loaded_count=4,
            artifact_verification_status="ALL_HASHES_VERIFIED",
            data_provenance="SYNTHETIC_V2_DATASET",
            timestamp=now_iso
        )
        data = res.model_dump() if hasattr(res, "model_dump") else (res.dict() if hasattr(res, "dict") else res.__dict__)
        assert data["service_status"] == "READY"
        assert data["model_version"] == "2.0.0"
        assert data["models_loaded_count"] == 4
        assert data["artifact_verification_status"] == "ALL_HASHES_VERIFIED"
        assert data["data_provenance"] == "SYNTHETIC_V2_DATASET"

    def test_model_registry_status_response_structure(self):
        """Verifies ModelRegistryStatusResponse structure and model status objects."""
        now_iso = datetime.utcnow().isoformat() + "Z"
        artifact_status = ModelArtifactStatus(
            model_id="model_a_classifier",
            model_name="LogisticRegression",
            model_version="2.0.0",
            status="LOADED",
            sha256_verified=True,
            data_provenance="SYNTHETIC_V2_DATASET"
        )
        data_artifact = artifact_status.model_dump() if hasattr(artifact_status, "model_dump") else (artifact_status.dict() if hasattr(artifact_status, "dict") else artifact_status.__dict__)

        res = ModelRegistryStatusResponse(
            service_status="READY",
            model_version="2.0.0",
            data_provenance="SYNTHETIC_V2_DATASET",
            timestamp=now_iso,
            models={"model_a_classifier": data_artifact}
        )
        data = res.model_dump() if hasattr(res, "model_dump") else (res.dict() if hasattr(res, "dict") else res.__dict__)
        assert data["service_status"] == "READY"
        assert data["data_provenance"] == "SYNTHETIC_V2_DATASET"
        assert "model_a_classifier" in data["models"]
        assert data["models"]["model_a_classifier"]["sha256_verified"] is True

    def test_registry_readiness_logic(self):
        """Verifies readiness calculation based on loaded models."""
        registry = ModelRegistry(models_dir=DEFAULT_MODELS_DIR)
        registry.initialize_registry()
        summary = registry.get_status_summary()

        assert summary["service_status"] == "READY"
        assert summary["models_loaded_count"] == 4
        assert summary["artifact_verification_status"] == "ALL_HASHES_VERIFIED"
        assert summary["data_provenance"] == "SYNTHETIC_V2_DATASET"

    def test_create_app_behavior(self):
        """Verifies FastAPI factory creates app if available, or raises informative error."""
        if not FASTAPI_AVAILABLE:
            with pytest.raises(RuntimeError) as exc_info:
                create_app()
            assert "FastAPI is not installed" in str(exc_info.value)
        else:
            app = create_app()
            assert app is not None
            assert app.title == "STS Fintech AI Financial Intelligence Microservice"
