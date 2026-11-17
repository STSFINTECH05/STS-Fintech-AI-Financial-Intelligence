"""
STS FINTECH AI FINANCIAL INTELLIGENCE
Model Integrity and Artifact Verification Tests (Milestone 2 - Step 7.4B-1)

Verifies:
1. Streaming SHA-256 calculation matches expected checksums for approved V2 models.
2. Registry loads exactly the 4 approved models (A, B, C, D).
3. Excluded models (Benchmark, Model E2) are never loaded into memory.
4. Deferred models (Model E1, E3) are quarantined and not loaded.
5. Corrupted / tampered artifacts trigger ModelIntegrityError.
6. Missing artifacts trigger FileNotFoundError.
7. Model status metadata contains no absolute filesystem paths and reports synthetic provenance.
"""

import os
import tempfile
import pytest
import joblib

from ml.service.config import DEFAULT_MODELS_DIR
from ml.service.model_loader import (
    ModelRegistry,
    ModelIntegrityError,
    APPROVED_MODEL_SPECS,
    compute_file_sha256
)


class TestModelIntegrity:
    """Test suite for SHA-256 model verification and registry loading."""

    def test_compute_file_sha256_matches_known_specs(self):
        """Verifies that all actual files in models/v2 match the hardcoded SHA-256 specifications."""
        for model_id, spec in APPROVED_MODEL_SPECS.items():
            filepath = os.path.join(DEFAULT_MODELS_DIR, spec["filename"])
            if os.path.exists(filepath):
                computed = compute_file_sha256(filepath)
                assert computed == spec["expected_sha256"], (
                    f"Checksum mismatch for {model_id} ({spec['filename']}): "
                    f"computed {computed} vs expected {spec['expected_sha256']}"
                )

    def test_registry_initialization_approved_models(self):
        """Verifies that initializing registry loads models A, B, C, D successfully."""
        registry = ModelRegistry(models_dir=DEFAULT_MODELS_DIR)
        status_info = registry.initialize_registry()

        # Exactly 4 models must be loaded in memory
        assert len(registry.loaded_models) == 4
        assert "model_a_classifier" in registry.loaded_models
        assert "model_b_anomaly_detector" in registry.loaded_models
        assert "model_c_forecaster" in registry.loaded_models
        assert "model_d_budget_predictor" in registry.loaded_models

        # Check status indicators
        for model_id in ["model_a_classifier", "model_b_anomaly_detector", "model_c_forecaster", "model_d_budget_predictor"]:
            assert status_info[model_id]["status"] == "LOADED"
            assert status_info[model_id]["sha256_verified"] is True
            assert status_info[model_id]["data_provenance"] == "SYNTHETIC_V2_DATASET"

    def test_excluded_models_are_quarantined_and_not_loaded(self):
        """Verifies that benchmark and rejected models are flagged EXCLUDED and never loaded."""
        registry = ModelRegistry(models_dir=DEFAULT_MODELS_DIR)
        status_info = registry.initialize_registry()

        # Model B supervised benchmark
        assert "model_b_supervised_benchmark" not in registry.loaded_models
        assert status_info["model_b_supervised_benchmark"]["status"] == "EXCLUDED"
        assert status_info["model_b_supervised_benchmark"]["sha256_verified"] is False

        # Model E2 late payment classifier
        assert "model_e2_late_payment" not in registry.loaded_models
        assert status_info["model_e2_late_payment"]["status"] == "EXCLUDED"
        assert status_info["model_e2_late_payment"]["sha256_verified"] is False

    def test_deferred_models_are_not_loaded(self):
        """Verifies that Model E1 and E3 remain DEFERRED and are not in memory."""
        registry = ModelRegistry(models_dir=DEFAULT_MODELS_DIR)
        status_info = registry.initialize_registry()

        # Model E1 and E3
        assert "model_e1_duplicate_matcher" not in registry.loaded_models
        assert "model_e3_vendor_screening" not in registry.loaded_models

        assert status_info["model_e1_duplicate_matcher"]["status"] == "DEFERRED"
        assert status_info["model_e3_vendor_screening"]["status"] == "DEFERRED"

    def test_tampered_model_raises_integrity_error(self):
        """Verifies that any modification to an artifact binary triggers ModelIntegrityError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a fake model artifact with invalid content
            fake_artifact_name = APPROVED_MODEL_SPECS["model_a_classifier"]["filename"]
            fake_path = os.path.join(tmp_dir, fake_artifact_name)
            with open(fake_path, "wb") as f:
                f.write(b"CORRUPTED_OR_TAMPERED_MODEL_PAYLOAD")

            registry = ModelRegistry(models_dir=tmp_dir)
            with pytest.raises(ModelIntegrityError) as exc_info:
                registry.initialize_registry()

            assert "Cryptographic hash mismatch" in str(exc_info.value)

    def test_missing_model_raises_file_not_found(self):
        """Verifies that a missing required model raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as empty_dir:
            registry = ModelRegistry(models_dir=empty_dir)
            with pytest.raises(FileNotFoundError) as exc_info:
                registry.initialize_registry()

            assert "Required model artifact not found on disk" in str(exc_info.value)

    def test_status_summary_security_and_provenance(self):
        """Verifies that status summary reports SYNTHETIC_V2_DATASET and does not leak absolute file paths."""
        registry = ModelRegistry(models_dir=DEFAULT_MODELS_DIR)
        registry.initialize_registry()
        summary = registry.get_status_summary()

        assert summary["service_status"] == "READY"
        assert summary["model_version"] == "2.0.0"
        assert summary["models_loaded_count"] == 4
        assert summary["artifact_verification_status"] == "ALL_HASHES_VERIFIED"
        assert summary["data_provenance"] == "SYNTHETIC_V2_DATASET"

        summary_str = str(summary)
        assert "/Users/" not in summary_str
        assert ".joblib" not in summary_str
