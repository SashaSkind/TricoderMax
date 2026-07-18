"""
Abstract detector module.

A module decides whether it applies to a study (metadata only — invariant 2) and
runs on the preprocessed volume, always returning a schema-valid `ModuleResult`.
The orchestrator guarantees `run()` failures become `status="failed"` results, but
modules should still catch their own internal errors where they can add detail.

`build_result()` centralizes attaching registry provenance (version, threshold,
calibration) so individual modules never hard-code calibration numbers (Flag F5).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Optional

import numpy as np

from src.contract import Evidence, FindingClass, ModuleResult, ModuleStatus
from src.registry import REGISTRY, ModuleDescriptor


class Module(ABC):
    module_id: str

    def __init__(self, module_id: str):
        self.module_id = module_id
        if module_id not in REGISTRY:
            raise KeyError(f"{module_id} not in registry")
        self.desc: ModuleDescriptor = REGISTRY[module_id]

    # ── to override ──────────────────────────────────────────────────────────
    @abstractmethod
    def applies_to(self, ctx) -> bool:  # ctx: StudyContext
        """Metadata-only decision. Never inspect pixels here (invariant 2)."""

    @abstractmethod
    def run(self, volume: np.ndarray, ctx) -> ModuleResult:
        """Run on the preprocessed HU/windowed volume → ModuleResult."""

    # ── helpers ──────────────────────────────────────────────────────────────
    def build_result(
        self,
        *,
        value: Optional[float],
        positive: Optional[bool] = None,
        detail: Optional[dict[str, float]] = None,
        evidence: Optional[Evidence] = None,
        runtime_ms: int = 0,
        status: ModuleStatus = "ok",
        error: Optional[str] = None,
        approximation: Optional[bool] = None,
    ) -> ModuleResult:
        d = self.desc
        return ModuleResult(
            module_id=d.module_id,
            module_version=d.version,
            finding_class=d.finding_class,
            result_type=d.result_type,  # type: ignore[arg-type]
            value=value,
            units=d.units,
            threshold=d.threshold,
            positive=positive,  # validator derives from value/threshold when None
            detail=detail or {},
            evidence=evidence or Evidence(),
            calibration=d.calibration,
            approximation=d.approximation if approximation is None else approximation,
            runtime_ms=runtime_ms,
            status=status,
            error=error,
        )

    def failed_result(self, error: str, runtime_ms: int = 0) -> ModuleResult:
        return self.build_result(
            value=None, positive=None, runtime_ms=runtime_ms, status="failed", error=error
        )

    def not_applicable_result(self, note: str = "") -> ModuleResult:
        return self.build_result(
            value=None,
            positive=None,
            evidence=Evidence(note=note or None),
            status="not_applicable",
        )


@contextmanager
def timed():
    """`with timed() as t: ...; t()` → elapsed ms after the block."""
    start = time.time()
    holder: dict[str, Any] = {}

    def elapsed() -> int:
        return int((holder.get("end", time.time()) - start) * 1000)

    try:
        yield elapsed
    finally:
        holder["end"] = time.time()
