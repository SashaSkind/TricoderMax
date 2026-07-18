"""
MockModule — canned, schema-valid results for any registered module.

This is the Phase-0 demo insurance: the entire worklist → panel → policy → UI
pipeline runs on these with zero ML installed. Values are deterministic per
(module_id, accession) so the demo is stable, and cover positive / negative /
approximation / failure paths depending on the accession.
"""

from __future__ import annotations

import hashlib

import numpy as np

from src.contract import Evidence, ModuleResult
from src.modules.base import Module, timed


def _h(*parts: str) -> float:
    """Deterministic pseudo-value in [0,1) from string parts."""
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return (int(digest[:8], 16) % 10_000) / 10_000.0


class MockModule(Module):
    """Stands in for any registered module id."""

    def applies_to(self, ctx) -> bool:
        return ctx.study_type in self.desc.applies_to or ctx.routing_ambiguous

    def run(self, volume: np.ndarray, ctx) -> ModuleResult:  # noqa: ARG002
        with timed() as t:
            acc = ctx.accession
            # Deterministic failure path for demoing failure isolation.
            if _h("fail", self.module_id, acc) > 0.92:
                return self.failed_result("mock: simulated module crash", runtime_ms=t())

            # Skew toward negative (r**2) so a realistic minority page — most
            # studies reorder, matching a real worklist's base rate.
            r = _h(self.module_id, acc) ** 2
            if self.desc.result_type == "measurement":
                thr = self.desc.threshold or 5.0
                value = round(r * thr * 2.2, 2)  # some above, some below cutoff
            else:
                value = round(r, 3)

            slices = sorted({int(r * 30) + k for k in (0, 1, 2)})
            evidence = Evidence(
                slice_indices=slices,
                overlay_paths=[],  # mocks write no pixel artifacts
                note="mock result — no real inference",
            )
            return self.build_result(value=value, evidence=evidence, runtime_ms=t())
