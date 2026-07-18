"""
Module registry — the catalog of panel members and the SINGLE SOURCE OF TRUTH
for calibration (Flag F5).

Each descriptor declares: which study types the module applies to, its finding
class, and its current calibration (from `eval/run_eval.py`). Modules attach the
calibration recorded here at result-build time rather than hard-coding AUCs, so
re-running eval updates one place.

`eval/run_eval.py` writes measured numbers back into `CALIBRATION` (persisted to
`registry_calibration.json`); this module loads that file if present.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.contract import Calibration, FindingClass, StudyType

_CALIB_FILE = Path(__file__).resolve().parent.parent / "registry_calibration.json"


@dataclass(frozen=True)
class ModuleDescriptor:
    module_id: str
    version: str
    finding_class: FindingClass
    applies_to: frozenset[StudyType]  # study types this module runs on
    result_type: str  # "probability" | "measurement"
    units: Optional[str] = None
    threshold: Optional[float] = None
    approximation: bool = False
    # Populated from registry_calibration.json when eval has run; None → uncalibrated.
    calibration: Optional[Calibration] = None


# Study types that any head-CT panel member applies to. `unknown` is included so
# ambiguous studies still trigger the module (invariant 4 — superset fan-out).
_HEAD_CT = frozenset(
    {StudyType.head_ct_noncontrast, StudyType.head_ct_other, StudyType.unknown}
)


def _load_calibration() -> dict[str, Calibration]:
    if not _CALIB_FILE.exists():
        return {}
    raw = json.loads(_CALIB_FILE.read_text())
    return {mid: Calibration(**c) for mid, c in raw.items()}


_CALIB = _load_calibration()


def _calib(module_id: str) -> Optional[Calibration]:
    return _CALIB.get(module_id)


# ── The catalog ──────────────────────────────────────────────────────────────
# Thresholds are the current operating points; calibration is attached from the
# eval file if present, else None (UI renders "uncalibrated").
REGISTRY: dict[str, ModuleDescriptor] = {
    "ich_v1": ModuleDescriptor(
        module_id="ich_v1",
        version="0.1.0",
        finding_class=FindingClass.intracranial_hemorrhage,
        applies_to=_HEAD_CT,
        result_type="probability",
        threshold=0.55,
        approximation=False,
        calibration=_calib("ich_v1"),
    ),
    "midline_shift_v1": ModuleDescriptor(
        module_id="midline_shift_v1",
        version="0.1.0",
        finding_class=FindingClass.midline_shift,
        applies_to=_HEAD_CT,
        result_type="measurement",
        units="mm",
        threshold=5.0,  # clinical cutoff
        approximation=True,  # crude proxy — must render visibly
        calibration=_calib("midline_shift_v1"),
    ),
    "hydrocephalus_v1": ModuleDescriptor(
        module_id="hydrocephalus_v1",
        version="0.1.0",
        finding_class=FindingClass.hydrocephalus,
        applies_to=_HEAD_CT,
        result_type="measurement",
        units="ratio",  # Evans' index proxy
        threshold=0.30,
        approximation=True,
        calibration=_calib("hydrocephalus_v1"),
    ),
}


def descriptors_for(study_type: StudyType) -> list[ModuleDescriptor]:
    """Modules that apply to a study type. Unknown → the full superset."""
    if study_type == StudyType.unknown:
        return list(REGISTRY.values())
    return [d for d in REGISTRY.values() if study_type in d.applies_to]


def all_descriptors() -> list[ModuleDescriptor]:
    return list(REGISTRY.values())
