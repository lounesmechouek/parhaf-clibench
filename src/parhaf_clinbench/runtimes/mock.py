"""Deterministic mock runtime for smoke tests and CPU CI."""

from __future__ import annotations

import json

from parhaf_clinbench.core.models import InferenceRequest
from parhaf_clinbench.data.canonicalize import canonical_to_dict
from parhaf_clinbench.runtimes.base import RuntimeBackend


class MockRuntime(RuntimeBackend):
    """Runtime that returns gold annotations when available."""

    @property
    def name(self) -> str:
        return "mock"

    @property
    def version(self) -> str:
        return "deterministic-v1"

    def infer(self, request: InferenceRequest) -> str:
        """Return deterministic canonical JSON for the provided request."""

        if request.gold is not None:
            return json.dumps(canonical_to_dict(request.gold), ensure_ascii=False)
        return json.dumps(
            {
                "document_id": request.document_id,
                "task": request.task.value,
                "speciality": None,
                "records": [],
            },
            ensure_ascii=False,
        )
