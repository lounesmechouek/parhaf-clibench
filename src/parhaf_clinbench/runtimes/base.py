"""Shared interface for inference runtime backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from parhaf_clinbench.core.models import InferenceRequest


class RuntimeBackend(ABC):
    """Minimal runtime backend contract."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Runtime name."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Runtime version."""

    @abstractmethod
    def infer(self, request: InferenceRequest) -> str:
        """Run inference and return raw JSON output."""

    def close(self) -> None:
        """Release runtime resources (optional implementation)."""

        return None
