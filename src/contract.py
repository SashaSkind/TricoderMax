"""
The module contract — the single stable seam between detector modules and the
Claude layer. Every panel member emits a `ModuleResult`; adding or swapping a
module never touches downstream code as long as this schema holds.

Design invariants enforced here (see README):
  - Every numeric output carries provenance (module + threshold; calibration or an
    explicit "uncalibrated" null).
  - A non-ok module can never masquerade as a positive finding.
  - `positive` is derived from value vs threshold (concern-upward convention).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FindingClass(str, Enum):
    intracranial_hemorrhage = "intracranial_hemorrhage"
    midline_shift = "midline_shift"
    hydrocephalus = "hydrocephalus"
    mass_effect = "mass_effect"
    calvarial_fracture = "calvarial_fracture"


# Finding classes whose presence can change immediate management → page-eligible.
# calvarial_fracture is intentionally excluded (reorder-only). Locked per F2.
MANAGEMENT_CHANGING: frozenset[FindingClass] = frozenset(
    {
        FindingClass.intracranial_hemorrhage,
        FindingClass.midline_shift,
        FindingClass.mass_effect,
        FindingClass.hydrocephalus,
    }
)

ResultType = Literal["probability", "measurement"]
ModuleStatus = Literal["ok", "failed", "not_applicable"]


class Calibration(BaseModel):
    """How a module's operating threshold was evaluated. Null → 'uncalibrated'."""

    model_config = ConfigDict(extra="forbid")

    eval_dataset: str  # e.g. "CQ500" — report the EXTERNAL set, never internal-only
    metric: str = "auc"
    metric_value: float
    n: int
    sensitivity_at_threshold: Optional[float] = None
    specificity_at_threshold: Optional[float] = None


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slice_indices: list[int] = Field(default_factory=list)
    overlay_paths: list[str] = Field(default_factory=list)  # under artifacts/ (gitignored)
    note: Optional[str] = None


class ModuleResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module_id: str
    module_version: str
    finding_class: FindingClass
    result_type: ResultType
    value: Optional[float] = None  # None ONLY when status != "ok"
    units: Optional[str] = None  # required iff result_type == "measurement"
    threshold: Optional[float] = None
    positive: Optional[bool] = None  # derived from value vs threshold
    detail: dict[str, float] = Field(default_factory=dict)
    evidence: Evidence = Field(default_factory=Evidence)
    calibration: Optional[Calibration] = None  # None → UI must render "uncalibrated"
    approximation: bool = False  # crude proxy → UI must render visibly
    runtime_ms: int = 0
    status: ModuleStatus = "ok"
    error: Optional[str] = None

    @model_validator(mode="after")
    def _enforce_invariants(self) -> "ModuleResult":
        if self.status == "ok":
            if self.value is None:
                raise ValueError("value required when status == 'ok'")
            if self.result_type == "probability" and not (0.0 <= self.value <= 1.0):
                raise ValueError(f"probability value {self.value} must be in [0, 1]")
            if self.result_type == "measurement" and not self.units:
                raise ValueError("measurement result requires 'units'")
            # Concern-upward convention: positive == value >= threshold (Flag F1).
            if self.threshold is not None:
                expected = self.value >= self.threshold
                if self.positive is None:
                    self.positive = expected
                elif self.positive != expected:
                    raise ValueError(
                        f"positive={self.positive} inconsistent with "
                        f"value={self.value} vs threshold={self.threshold}"
                    )
        else:
            if self.positive:
                raise ValueError("a non-ok module result cannot be 'positive'")
            if self.status == "failed" and not self.error:
                raise ValueError("failed status requires an 'error' message")
        return self

    @property
    def is_management_changing(self) -> bool:
        return self.finding_class in MANAGEMENT_CHANGING


class StudyType(str, Enum):
    head_ct_noncontrast = "head_ct_noncontrast"
    head_ct_other = "head_ct_other"
    unknown = "unknown"  # → orchestrator runs the SUPERSET of panels (invariant 4)


class StudyContext(BaseModel):
    """Study metadata only — never pixels (invariant 2). Routing consumes this."""

    model_config = ConfigDict(extra="forbid")

    accession: str  # correlation id used across all structured logs
    study_uid: str
    study_type: StudyType = StudyType.unknown
    modality: str = "CT"
    n_slices: int
    age: Optional[int] = None
    sex: Optional[Literal["M", "F", "O"]] = None
    indication: Optional[str] = None  # requisition / reason-for-exam text
    acquired_at: Optional[datetime] = None
    prior_report: Optional[str] = None
    volume_path: Optional[str] = None  # ref to preprocessed volume on disk, not pixels
    dicom_meta: dict[str, str] = Field(default_factory=dict)
    routing_ambiguous: bool = False  # True when the superset was run


class PanelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    study: StudyContext
    results: list[ModuleResult]
    modules_selected: list[str]  # module_ids routing chose to run
    modules_failed: list[str] = Field(default_factory=list)
    superset_run: bool = False  # invariant 4 fan-out actually happened
    started_at: datetime
    completed_at: datetime
    total_runtime_ms: int

    @model_validator(mode="after")
    def _integrity(self) -> "PanelOutput":
        got = {r.module_id for r in self.results}
        missing = set(self.modules_selected) - got
        if missing:  # a selected module vanishing would be a silently skipped finding
            raise ValueError(f"selected modules missing from results: {sorted(missing)}")
        declared = set(self.modules_failed)
        actual = {r.module_id for r in self.results if r.status == "failed"}
        if declared != actual:
            raise ValueError(
                f"modules_failed {sorted(declared)} != actual failures {sorted(actual)}"
            )
        return self

    @property
    def positive_results(self) -> list[ModuleResult]:
        return [r for r in self.results if r.status == "ok" and r.positive]
