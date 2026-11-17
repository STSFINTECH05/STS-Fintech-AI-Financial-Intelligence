"""
STS FINTECH AI FINANCIAL INTELLIGENCE
Microservice Configuration Module (Milestone 2 - Step 7.4B-1)

Loads runtime configuration from environment variables with safe defaults.
Never exposes secret values or hardcoded credentials.
"""

import os
from typing import Optional
from dataclasses import dataclass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODELS_DIR = os.path.join(BASE_DIR, "ml", "models", "v2")


@dataclass(frozen=True)
class ServiceConfig:
    """Immutable microservice configuration."""

    host: str = os.getenv("STS_ML_HOST", "127.0.0.1")
    port: int = int(os.getenv("STS_ML_PORT", "8000"))
    environment: str = os.getenv("STS_ML_ENV", "development")
    models_dir: str = os.getenv("STS_ML_MODELS_DIR", DEFAULT_MODELS_DIR)
    replay_window_seconds: int = int(os.getenv("STS_ML_REPLAY_WINDOW", "300"))

    # Internal service keys are accessed only via dedicated helper methods
    _active_key: Optional[str] = os.getenv("STS_INTERNAL_SERVICE_KEY")
    _previous_key: Optional[str] = os.getenv("STS_INTERNAL_SERVICE_KEY_PREVIOUS")

    @property
    def has_configured_key(self) -> bool:
        """Indicates whether the internal service key has been set."""
        key = self.get_active_key()
        return bool(key and len(key.strip()) >= 32)

    def get_active_key(self) -> Optional[str]:
        """Returns the active internal service key."""
        return os.getenv("STS_INTERNAL_SERVICE_KEY") or self._active_key

    def get_previous_key(self) -> Optional[str]:
        """Returns the fallback key for zero-downtime key rotation."""
        return os.getenv("STS_INTERNAL_SERVICE_KEY_PREVIOUS") or self._previous_key


config = ServiceConfig()
