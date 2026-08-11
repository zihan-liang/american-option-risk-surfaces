"""Contracts and utilities for the independent-method evaluation campaign."""

from .records import MethodRunRecord, RunPhase, RunStatus
from .registry import METHOD_REGISTRY, MethodCategory, MethodSpec, get_method_spec

__all__ = [
    "METHOD_REGISTRY",
    "MethodCategory",
    "MethodRunRecord",
    "MethodSpec",
    "RunPhase",
    "RunStatus",
    "get_method_spec",
]
