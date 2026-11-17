"""
STS FINTECH AI FINANCIAL INTELLIGENCE
Model Registry and Integrity Verification Loader (Milestone 2 - Step 7.4B-1)

Loads only approved V2 model artifacts after strict cryptographic SHA-256 validation.
Bars unapproved benchmarks and rejected models from memory.
Never exposes absolute filesystem paths in client metadata.
"""

import os
import hashlib
import logging
from typing import Dict, Any, Optional
import joblib

logger = logging.getLogger("sts_ml_service.model_loader")

# Master Specification of Approved and Quarantined Artifacts
APPROVED_MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "model_a_classifier": {
        "model_id": "model_a_classifier",
        "model_name": "LogisticRegression",
        "model_version": "2.0.0",
        "filename": "expense_classifier_v2.joblib",
        "expected_sha256": "6b1405b3770ff0ae55c5de8871cf347e80f4e6201472a1ca5e4b08df44e4d1d2",
        "target_status": "LOADED",
        "category": "APPROVED_FOR_PILOT"
    },
    "model_b_anomaly_detector": {
        "model_id": "model_b_anomaly_detector",
        "model_name": "IsolationForestV2AnomalyDetector",
        "model_version": "2.0.0",
        "filename": "anomaly_detector_v2.joblib",
        "expected_sha256": "e7094d751b337862bc6892c551f215dea0957a1b7e1926a851f7425837c22f29",
        "target_status": "LOADED",
        "category": "APPROVED_FOR_SCREENING"
    },
    "model_c_forecaster": {
        "model_id": "model_c_forecaster",
        "model_name": "RandomForestRegressor",
        "model_version": "2.0.0",
        "filename": "expense_forecaster_v2.joblib",
        "expected_sha256": "ea71b872864f3ab8dba53ab96422e24c6d21b730c9fee0191b9fded950de6b22",
        "target_status": "LOADED",
        "category": "APPROVED_FOR_PILOT"
    },
    "model_d_budget_predictor": {
        "model_id": "model_d_budget_predictor",
        "model_name": "RandomForestRegressor",
        "model_version": "2.0.0",
        "filename": "budget_predictor_v2.joblib",
        "expected_sha256": "3a56c7c086b01814fdce1367e967292122c9ad2507bae1023c9a26e1dc8e3199",
        "target_status": "LOADED",
        "category": "APPROVED_FOR_PILOT"
    },
    "model_e1_duplicate_matcher": {
        "model_id": "model_e1_duplicate_matcher",
        "model_name": "OperationalDuplicateMatcher",
        "model_version": "2.0.0",
        "filename": "vendor_duplicate_detector_v2.joblib",
        "expected_sha256": "0c4156f335a8ea0d64fdb41d7406698e17b28959805ae736742c9cd38d0e8f23",
        "target_status": "DEFERRED",
        "category": "AWAITING_CONTRACT_VERIFICATION"
    },
    "model_e3_vendor_screening": {
        "model_id": "model_e3_vendor_screening",
        "model_name": "EnterpriseMultiFactorRuleScreeningEngineV2",
        "model_version": "2.0.0",
        "filename": "vendor_screening_v2.joblib",
        "expected_sha256": "b81b393464ed0577734a38e59227b811184302ecd1f756b67677400bddcc6379",
        "target_status": "DEFERRED",
        "category": "AWAITING_CONTRACT_VERIFICATION"
    },
    "model_b_supervised_benchmark": {
        "model_id": "model_b_supervised_benchmark",
        "model_name": "RandomForestClassifier",
        "model_version": "2.0.0",
        "filename": "anomaly_supervised_benchmark_v2.joblib",
        "expected_sha256": "7fb574b12019b65fcbcbca8cf5bdf9b0b896723a67d1c86dd3a191080f1c3c59",
        "target_status": "EXCLUDED",
        "category": "EVALUATION_BENCHMARK_ONLY"
    },
    "model_e2_late_payment": {
        "model_id": "model_e2_late_payment",
        "model_name": "GradientBoostingClassifier",
        "model_version": "2.0.0",
        "filename": "vendor_payment_predictor_v2.joblib",
        "expected_sha256": "afa6ed2fbd1470b31eb9fadd8f4ddc94b5c68d5925b6f670227bfba77401d149",
        "target_status": "EXCLUDED",
        "category": "REJECTED_INSUFFICIENT_DISCRIMINATION"
    }
}


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash in streaming 8KB chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class ModelIntegrityError(Exception):
    """Raised when an artifact's cryptographic checksum does not match verified specification."""
    pass


class ModelRegistry:
    """
    Central in-memory model registry managing approved V2 ML artifacts.
    Enforces cryptographic SHA-256 verification before loading any binary.
    """

    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.loaded_models: Dict[str, Any] = {}
        self.model_status_info: Dict[str, Dict[str, Any]] = {}

    def initialize_registry(self) -> Dict[str, Any]:
        """
        Validates all approved model files, loads permitted artifacts into memory,
        and records registration status for every specification entry.
        """
        for model_id, spec in APPROVED_MODEL_SPECS.items():
            filename = spec["filename"]
            target_status = spec["target_status"]
            filepath = os.path.join(self.models_dir, filename)

            # 1. Handle Excluded Models
            if target_status == "EXCLUDED":
                self.model_status_info[model_id] = {
                    "model_id": model_id,
                    "model_name": spec["model_name"],
                    "model_version": spec["model_version"],
                    "status": "EXCLUDED",
                    "sha256_verified": False,
                    "category": spec["category"],
                    "data_provenance": "SYNTHETIC_V2_DATASET"
                }
                continue

            # 2. Handle Deferred Models
            if target_status == "DEFERRED":
                is_valid = False
                if os.path.exists(filepath):
                    actual_hash = compute_file_sha256(filepath)
                    is_valid = (actual_hash == spec["expected_sha256"])

                self.model_status_info[model_id] = {
                    "model_id": model_id,
                    "model_name": spec["model_name"],
                    "model_version": spec["model_version"],
                    "status": "DEFERRED",
                    "sha256_verified": is_valid,
                    "category": spec["category"],
                    "data_provenance": "SYNTHETIC_V2_DATASET"
                }
                continue

            # 3. Handle Approved Active Models (LOADED)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Required model artifact not found on disk: {filename}")

            actual_hash = compute_file_sha256(filepath)
            expected_hash = spec["expected_sha256"]

            if actual_hash != expected_hash:
                error_msg = (
                    f"Cryptographic hash mismatch for {model_id} ({filename})! "
                    f"Expected: {expected_hash}, Computed: {actual_hash}. "
                    "Refusing to load untrusted or corrupted binary."
                )
                logger.error(error_msg)
                raise ModelIntegrityError(error_msg)

            # Safely load verified model into memory
            try:
                loaded_obj = joblib.load(filepath)
                self.loaded_models[model_id] = loaded_obj
                self.model_status_info[model_id] = {
                    "model_id": model_id,
                    "model_name": spec["model_name"],
                    "model_version": spec["model_version"],
                    "status": "LOADED",
                    "sha256_verified": True,
                    "category": spec["category"],
                    "data_provenance": "SYNTHETIC_V2_DATASET"
                }
                logger.info("Successfully verified and loaded model: %s (%s)", model_id, filename)
            except Exception as e:
                logger.error("Failed to unpickle verified artifact %s: %s", filename, str(e))
                raise e

        return self.model_status_info

    def get_model(self, model_id: str) -> Any:
        """Retrieves a loaded model instance from memory."""
        if model_id not in self.loaded_models:
            raise KeyError(f"Model {model_id} is not loaded in the active registry.")
        return self.loaded_models[model_id]

    def get_status_summary(self) -> Dict[str, Any]:
        """Returns safe model status metadata for health endpoints."""
        return {
            "service_status": "READY" if len(self.loaded_models) >= 4 else "DEGRADED",
            "model_version": "2.0.0",
            "models_loaded_count": len(self.loaded_models),
            "artifact_verification_status": "ALL_HASHES_VERIFIED" if all(
                s.get("sha256_verified") for s in self.model_status_info.values() if s.get("status") == "LOADED"
            ) else "VERIFICATION_FAILURE",
            "data_provenance": "SYNTHETIC_V2_DATASET",
            "models": self.model_status_info
        }
