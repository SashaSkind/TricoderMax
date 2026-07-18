"""
Hydrocephalus module (hydrocephalus_v1) — STUB.

Per the accepted scope decision (Flag F3), hydrocephalus has no build phase yet.
Rather than fabricate a value in real mode, this stub reports `not_applicable` so
the panel stays complete and honest. A future phase can implement the Evans'-index
proxy the registry already reserves (approximation, ratio, threshold 0.30).
"""

from __future__ import annotations

import numpy as np

from src.contract import ModuleResult
from src.modules.base import Module


class HydrocephalusModule(Module):
    def __init__(self, module_id: str = "hydrocephalus_v1"):
        super().__init__(module_id)

    def applies_to(self, ctx) -> bool:  # noqa: ARG002
        return False  # not implemented yet → never claims to apply

    def run(self, volume: np.ndarray, ctx) -> ModuleResult:  # noqa: ARG002
        return self.not_applicable_result("hydrocephalus module not implemented (stub)")
