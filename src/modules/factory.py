"""
Module factory — maps a registry id to a concrete Module instance.

When `config.USE_MOCKS` is set (Phase 0 default), every id resolves to MockModule.
Otherwise real modules are used where implemented, falling back to MockModule for
ids not yet built so the panel is always complete.
"""

from __future__ import annotations

from src import config
from src.modules.base import Module
from src.modules.mock import MockModule

# Real modules register here as phases land.
_REAL: dict[str, type[Module]] = {}


def _load_real() -> None:
    """Import real modules lazily so Phase-0/mock runs need no torch/scipy."""
    if _REAL:
        return
    try:
        from src.modules.ich import ICHModule

        _REAL["ich_v1"] = ICHModule
    except Exception:  # noqa: BLE001 — optional heavy deps may be absent
        pass
    try:
        from src.modules.midline_shift import MidlineShiftModule

        _REAL["midline_shift_v1"] = MidlineShiftModule
    except Exception:  # noqa: BLE001
        pass
    try:
        from src.modules.hydrocephalus import HydrocephalusModule

        _REAL["hydrocephalus_v1"] = HydrocephalusModule
    except Exception:  # noqa: BLE001
        pass


def build_module(module_id: str) -> Module:
    if config.USE_MOCKS:
        return MockModule(module_id)
    _load_real()
    cls = _REAL.get(module_id, MockModule)
    return cls(module_id)
