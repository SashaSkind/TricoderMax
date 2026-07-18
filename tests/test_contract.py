"""Contract invariants — the seam every module must honor."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from src.contract import (
    Calibration,
    Evidence,
    FindingClass,
    ModuleResult,
    PanelOutput,
    StudyContext,
    StudyType,
)
from src.modules.factory import build_module
from src.registry import all_descriptors


def test_probability_out_of_range_rejected():
    with pytest.raises(ValidationError):
        ModuleResult(
            module_id="ich_v1", module_version="0.1.0",
            finding_class=FindingClass.intracranial_hemorrhage,
            result_type="probability", value=1.5, threshold=0.5,
        )


def test_positive_derived_from_threshold():
    r = ModuleResult(
        module_id="ich_v1", module_version="0.1.0",
        finding_class=FindingClass.intracranial_hemorrhage,
        result_type="probability", value=0.8, threshold=0.55,
    )
    assert r.positive is True
    r2 = ModuleResult(
        module_id="ich_v1", module_version="0.1.0",
        finding_class=FindingClass.intracranial_hemorrhage,
        result_type="probability", value=0.2, threshold=0.55,
    )
    assert r2.positive is False


def test_positive_inconsistent_rejected():
    with pytest.raises(ValidationError):
        ModuleResult(
            module_id="ich_v1", module_version="0.1.0",
            finding_class=FindingClass.intracranial_hemorrhage,
            result_type="probability", value=0.9, threshold=0.55, positive=False,
        )


def test_measurement_requires_units():
    with pytest.raises(ValidationError):
        ModuleResult(
            module_id="midline_shift_v1", module_version="0.1.0",
            finding_class=FindingClass.midline_shift,
            result_type="measurement", value=7.2, threshold=5.0,  # no units
        )


def test_failed_status_cannot_be_positive_and_needs_error():
    with pytest.raises(ValidationError):
        ModuleResult(
            module_id="ich_v1", module_version="0.1.0",
            finding_class=FindingClass.intracranial_hemorrhage,
            result_type="probability", status="failed", positive=True, error="x",
        )
    with pytest.raises(ValidationError):
        ModuleResult(
            module_id="ich_v1", module_version="0.1.0",
            finding_class=FindingClass.intracranial_hemorrhage,
            result_type="probability", status="failed",  # no error message
        )


def test_management_changing_set():
    r = ModuleResult(
        module_id="ich_v1", module_version="0.1.0",
        finding_class=FindingClass.intracranial_hemorrhage,
        result_type="probability", value=0.8, threshold=0.55,
    )
    assert r.is_management_changing is True
    frac = ModuleResult(
        module_id="frac", module_version="0.1.0",
        finding_class=FindingClass.calvarial_fracture,
        result_type="probability", value=0.8, threshold=0.55,
    )
    assert frac.is_management_changing is False


def test_panel_output_integrity_catches_missing_module():
    from datetime import datetime, timezone

    ctx = StudyContext(accession="A", study_uid="1", n_slices=10)
    with pytest.raises(ValidationError):
        PanelOutput(
            study=ctx, results=[], modules_selected=["ich_v1"],
            started_at=datetime.now(timezone.utc), completed_at=datetime.now(timezone.utc),
            total_runtime_ms=1,
        )


@pytest.mark.parametrize("desc", all_descriptors(), ids=lambda d: d.module_id)
def test_every_module_returns_valid_contract_on_synthetic_volume(desc):
    """Invariant: every registered module emits a schema-valid ModuleResult."""
    ctx = StudyContext(
        accession="SYN-1", study_uid="1.2.3", study_type=StudyType.head_ct_noncontrast, n_slices=16
    )
    vol = np.random.default_rng(0).normal(30, 20, size=(16, 64, 64)).astype(np.float32)
    module = build_module(desc.module_id)  # mocks in test env
    result = module.run(vol, ctx)
    assert isinstance(result, ModuleResult)
    assert result.module_id == desc.module_id
    assert result.finding_class == desc.finding_class
