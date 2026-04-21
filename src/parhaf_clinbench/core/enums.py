"""Core enumerations used across PARHAF-CLINBENCH."""

from __future__ import annotations

from enum import StrEnum


class TaskId(StrEnum):
    """Supported task identifiers for benchmark v1."""

    PSEUDO = "pseudo"
    INFECTIO = "infectio"
    RESPONSE = "response"
    SCENARIO = "scenario"


class TrackId(StrEnum):
    """Supported evaluation tracks."""

    ZEROSHOT = "zero-shot"
    FEWSHOT = "few-shot"


class RuntimeName(StrEnum):
    """Supported runtime backends."""

    MOCK = "mock"
    VLLM = "vllm"
    GLINER = "gliner"
