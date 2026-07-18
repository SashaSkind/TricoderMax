"""
Claude layer — runs strictly DOWNSTREAM of the detector panel (invariant 2).

It aggregates all ModuleResults + clinical context, verifies each detector claim
against context, and produces a priority ranking. It NEVER decides which detectors
run and NEVER removes a study from the queue — the worst it can do is decline to
escalate, which still leaves the study in the reading worklist.

Phase 0 ships the schema + the deterministic detector-only fallback. Phase 5 adds
the real Claude call; on any parse failure or `abstain: true` it falls back here.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.contract import PanelOutput

PriorityBand = Literal["immediate", "urgent", "routine"]


class Verification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    module_id: str
    supported: bool
    reasoning: str


class ClaudeAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    priority_score: float = Field(ge=0.0, le=1.0)
    priority_band: PriorityBand
    verification: list[Verification] = Field(default_factory=list)
    summary: str
    caveats: list[str] = Field(default_factory=list)
    abstain: bool = False
    source: Literal["claude", "fallback"] = "claude"

    def contradicts(self, module_id: str) -> bool:
        """True if Claude explicitly did NOT support this module's finding."""
        for v in self.verification:
            if v.module_id == module_id:
                return not v.supported
        return False


def _band_from_score(score: float) -> PriorityBand:
    if score >= 0.75:
        return "immediate"
    if score >= 0.40:
        return "urgent"
    return "routine"


def detector_only_assessment(panel: PanelOutput) -> ClaudeAssessment:
    """Deterministic fallback: rank from detector probabilities alone.

    Priority = the strongest positive management-changing finding, else the
    strongest signal present. Every positive module is marked 'supported' since we
    have no context to contradict it — the conservative choice (never suppress).
    """
    positives = panel.positive_results
    mgmt_positive = [r for r in positives if r.is_management_changing]

    def signal(r) -> float:
        # Probabilities are already [0,1]; measurements → fraction over threshold.
        if r.result_type == "probability":
            return float(r.value or 0.0)
        if r.threshold:
            return min(1.0, float(r.value or 0.0) / (2.0 * r.threshold))
        return 0.5 if r.positive else 0.0

    if mgmt_positive:
        score = max(signal(r) for r in mgmt_positive)
    elif positives:
        score = 0.5 * max(signal(r) for r in positives)
    else:
        score = 0.0

    verification = [
        Verification(
            module_id=r.module_id,
            supported=True,
            reasoning="detector-only fallback: no clinical context available to verify",
        )
        for r in positives
    ]
    failed = panel.modules_failed
    caveats = ["Detector-only ranking (Claude layer not invoked)."]
    if failed:
        caveats.append(f"Modules failed and were not considered: {', '.join(failed)}.")

    labels = ", ".join(sorted({r.finding_class.value for r in positives})) or "no positive findings"
    return ClaudeAssessment(
        priority_score=round(score, 3),
        priority_band=_band_from_score(score),
        verification=verification,
        summary=f"Detector panel flagged: {labels}.",
        caveats=caveats,
        abstain=False,
        source="fallback",
    )


def assess(panel: PanelOutput, prior_report: Optional[str] = None) -> ClaudeAssessment:
    """Phase-0/1 entry point: fallback only. Phase 5 overrides with the real call."""
    return detector_only_assessment(panel)
