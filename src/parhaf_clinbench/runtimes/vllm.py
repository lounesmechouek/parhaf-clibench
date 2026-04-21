"""Runtime vLLM OpenAI-compatible."""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

from parhaf_clinbench.chunking.merger import merge_canonical_documents
from parhaf_clinbench.chunking.splitter import make_chunks
from parhaf_clinbench.core.enums import TaskId
from parhaf_clinbench.core.models import (
    INFECTIO_LABELS,
    INFECTIO_NEGATIONS,
    PSEUDO_LABELS,
    RESPONSE_LABELS,
    SCENARIO_FIELDS,
    SCENARIO_SPECIALITIES,
    CanonicalDocument,
    InferenceRequest,
)
from parhaf_clinbench.runtimes.base import RuntimeBackend

_LOG = logging.getLogger(__name__)


class _WordCountTokenizer:
    """Minimal tokenizer fallback: one token per whitespace-separated word."""

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[str]:
        return text.split()

_BEGIN_MARKER = "<<<BEGIN_TEXT>>>"
_END_MARKER = "<<<END_TEXT>>>"

_CHUNK_THRESHOLD = 0.9

_CHUNK_TARGET = 0.85

_DEFAULT_MAX_NEW_TOKENS = 1024


def _replace_text_in_prompt(prompt: str, new_text: str) -> str:
    """Replace text enclosed by BEGIN/END markers in a rendered prompt."""
    b = prompt.index(_BEGIN_MARKER) + len(_BEGIN_MARKER)
    e = prompt.index(_END_MARKER)
    return prompt[:b] + "\n" + new_text + "\n" + prompt[e:]


def _record_schema(task: TaskId) -> dict[str, Any]:
    """Return strict JSON Schema for one canonical record."""

    label_values: list[str]
    attributes_schema: dict[str, Any]

    if task == TaskId.PSEUDO:
        label_values = sorted(PSEUDO_LABELS)
        attributes_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "role": {
                    "type": "string",
                    "enum": ["Carer", "Other", "Patient"],
                }
            },
        }
    elif task == TaskId.INFECTIO:
        label_values = sorted(INFECTIO_LABELS)
        attributes_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "negation": {
                    "type": "string",
                    "enum": sorted(INFECTIO_NEGATIONS),
                }
            },
            "required": ["negation"],
        }
    elif task == TaskId.RESPONSE:
        label_values = sorted(RESPONSE_LABELS)
        attributes_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        }
    else:
        label_values = sorted(SCENARIO_FIELDS)
        attributes_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        }

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "label": {"type": "string", "enum": label_values},
            "text": {"type": "string"},
            "start": {"type": "integer", "minimum": 0},
            "end": {"type": "integer", "minimum": 0},
            "attributes": attributes_schema,
        },
        "required": ["label", "text", "start", "end", "attributes"],
    }


def _response_format_for_task(task: TaskId) -> dict[str, Any]:
    """Build strict structured-output response format for a task."""

    if task == TaskId.SCENARIO:
        # NOTE: Scenario requires `speciality` with constrained allowed values.
        extra_properties: dict[str, Any] = {
            "speciality": {
                "type": "string",
                "enum": sorted(SCENARIO_SPECIALITIES),
            }
        }
        required_fields = ["document_id", "task", "speciality", "records"]
    else:
        # NOTE: Other tasks do not require `speciality` in strict schema.
        extra_properties = {}
        required_fields = ["document_id", "task", "records"]

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "document_id": {"type": "string", "minLength": 1},
            "task": {"type": "string", "const": task.value},
            **extra_properties,
            "records": {
                "type": "array",
                "items": _record_schema(task),
            },
        },
        "required": required_fields,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"parhaf_{task.value}_canonical_document",
            "strict": True,
            "schema": schema,
        },
    }


class VllmRuntime(RuntimeBackend):
    """vLLM runtime adapter exposing an OpenAI-compatible interface."""

    def __init__(
        self,
        *,
        api_base: str,
        model_hf_id: str,
        timeout_seconds: int = 120,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
        seed: int | None = None,
        max_context_tokens: int = 131072,
        tokenizer_revision: str = "main",
    ) -> None:
        self._api_base = api_base.rstrip("/")
        self._model_hf_id = model_hf_id
        self._timeout_seconds = timeout_seconds
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._seed = seed
        self._max_context_tokens = max_context_tokens
        self._tokenizer_revision = tokenizer_revision
        self._tokenizer: Any = None  # NOTE: Lazy tokenizer initialization.

    @property
    def name(self) -> str:
        return "vllm"

    @property
    def version(self) -> str:
        return "openai-compatible"

    def _get_tokenizer(self) -> Any:
        """Load tokenizer lazily and fallback to word-based counting."""

        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._model_hf_id,
                    revision=self._tokenizer_revision,
                    trust_remote_code=True,
                )
            except Exception as exc:
                _LOG.warning(
                    "Failed to load tokenizer for %s: %s. "
                    "Falling back to word-count approximation.",
                    self._model_hf_id,
                    exc,
                )
                self._tokenizer = _WordCountTokenizer()
        return self._tokenizer

    def _count_tokens(self, text: str) -> int:
        tok = self._get_tokenizer()
        return len(tok.encode(text, add_special_tokens=False))

    def _call_api(self, prompt: str, task: TaskId) -> str:
        """Call vLLM chat API and return raw JSON content string."""
        payload: dict[str, Any] = {
            "model": self._model_hf_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
            "top_p": self._top_p,
            "response_format": _response_format_for_task(task),
        }
        if self._max_tokens is not None:
            payload["max_tokens"] = self._max_tokens
        if self._seed is not None:
            payload["seed"] = self._seed
        response = requests.post(
            f"{self._api_base}/chat/completions",
            json=payload,
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices", [])
        if not choices:
            raise RuntimeError("vLLM response contains no generation choices")
        content = choices[0].get("message", {}).get("content", "")
        if not isinstance(content, str):
            raise RuntimeError("Unexpected vLLM response format")
        stripped = content.strip()
        if stripped.startswith("```"):
            # NOTE: Strict JSON schema output should never include markdown fences.
            _LOG.warning(
                "vLLM response contains markdown fences despite strict JSON schema "
                "(model=%s, task=%s). Check the model's structured-output conformance.",
                self._model_hf_id,
                task.value,
            )
            stripped = stripped.lstrip("`")
            if stripped.startswith("json"):
                stripped = stripped[4:]
            stripped = stripped.rstrip("`").strip()
        return stripped

    def infer(self, request: InferenceRequest) -> str:
        """Run single-pass inference, enabling chunking when context is near limit."""

        effective_max_new = self._max_tokens or _DEFAULT_MAX_NEW_TOKENS
        prompt_tokens = self._count_tokens(request.prompt)
        if prompt_tokens + effective_max_new >= _CHUNK_THRESHOLD * self._max_context_tokens:
            _LOG.info(
                "Chunking enabled: prompt=%d tokens + max_new=%d >= %.0f%% of %d "
                "(document_id=%s, task=%s)",
                prompt_tokens,
                effective_max_new,
                _CHUNK_THRESHOLD * 100,
                self._max_context_tokens,
                request.document_id,
                request.task.value,
            )
            return self._chunk_infer(request, effective_max_new)
        return self._call_api(request.prompt, request.task)

    def _chunk_infer(self, request: InferenceRequest, effective_max_new: int) -> str:
        """Chunk prompt text, infer per chunk, and merge canonical outputs."""
        tokenizer = self._get_tokenizer()
        overhead = self._count_tokens(_replace_text_in_prompt(request.prompt, ""))
        text_budget = int(_CHUNK_TARGET * self._max_context_tokens) - effective_max_new - overhead
        if text_budget <= 0:
            _LOG.warning(
                "Negative text budget after subtracting overhead+max_new "
                "(overhead=%d, max_new=%d, max_context=%d). "
                "Sending full prompt without chunking.",
                overhead,
                effective_max_new,
                self._max_context_tokens,
            )
            return self._call_api(request.prompt, request.task)

        chunks = make_chunks(request.text, tokenizer, text_budget)
        chunk_results: list[tuple[CanonicalDocument, int]] = []
        for chunk in chunks:
            chunk_prompt = _replace_text_in_prompt(request.prompt, chunk.text)
            raw = self._call_api(chunk_prompt, request.task)
            try:
                doc = CanonicalDocument.model_validate(json.loads(raw))
                chunk_results.append((doc, chunk.start_char))
            except Exception as exc:
                _LOG.warning(
                    "Chunk skipped (parse error) for document_id=%s: %s",
                    request.document_id,
                    exc,
                )

        if not chunk_results:
            return ""

        merged = merge_canonical_documents(chunk_results)
        return json.dumps(merged.model_dump(mode="json"), ensure_ascii=False)
