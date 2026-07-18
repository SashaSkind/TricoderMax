"""Policy: deterministic page/reorder split and its three page clauses."""

from __future__ import annotations

from datetime import datetime, timezone

from src.claude_layer import ClaudeAssessment, Verification, detector_only_assessment
from src.contract import (
    Evidence,
    FindingClass,
    ModuleResult,
    PanelOutput,
    StudyContext,
)
from src.policy import decide


def _panel(results: list[ModuleResult]) -> PanelOutput:
    ctx = StudyContext(accession="P-1", study_uid="1", n_slices=10)
    now = datetime.now(timezone.utc)
    selected = [r.module_id for r in results]
    failed = [r.module_id for r in results if r.status == "failed"]
    return PanelOutput(
        study=ctx, results=results, modules_selected=selected, modules_failed=failed,
        started_at=now, completed_at=now, total_runtime_ms=1,
    )


def _ich(value: float) -> ModuleResult:
    return ModuleResult(
        module_id="ich_v1", module_version="0.1.0",
        finding_class=FindingClass.intracranial_hemorrhage,
        result_type="probability", value=value, threshold=0.55,
    )


def _fracture(value: float) -> ModuleResult:
    return ModuleResult(
        module_id="frac_v1", module_version="0.1.0",
        finding_class=FindingClass.calvarial_fracture,
        result_type="probability", value=value, threshold=0.5,
    )


def test_positive_management_changing_not_contradicted_pages():
    panel = _panel([_ich(0.9)])
    a = detector_only_assessment(panel)
    d = decide(panel, a)
    assert d.action == "page" and "ich_v1" in d.paging_findings


def test_positive_but_not_management_changing_reorders():
    panel = _panel([_fracture(0.9)])  # calvarial fracture is reorder-only
    a = detector_only_assessment(panel)
    d = decide(panel, a)
    assert d.action == "reorder"


def test_claude_contradiction_blocks_page():
    panel = _panel([_ich(0.9)])
    a = ClaudeAssessment(
        priority_score=0.2, priority_band="routine",
        verification=[Verification(module_id="ich_v1", supported=False, reasoning="skull-base partial volume")],
        summary="down-weighted", abstain=False, source="claude",
    )
    d = decide(panel, a)
    assert d.action == "reorder"
    assert any("contradicts" in line for line in d.audit)


def test_negative_panel_reorders():
    panel = _panel([_ich(0.1)])
    a = detector_only_assessment(panel)
    d = decide(panel, a)
    assert d.action == "reorder" and d.paging_findings == []


def test_failed_module_is_surfaced_not_blocking():
    failed = ModuleResult(
        module_id="ich_v1", module_version="0.1.0",
        finding_class=FindingClass.intracranial_hemorrhage,
        result_type="probability", status="failed", error="boom",
    )
    panel = _panel([failed])
    a = detector_only_assessment(panel)
    d = decide(panel, a)
    assert d.action == "reorder"
    assert any("failed" in line for line in d.audit)
