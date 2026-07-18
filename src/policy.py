"""
Policy — the deterministic page/reorder split. NOT a model.

A study is PAGED (immediate radiologist attention) iff ALL hold:
  1. at least one detector is positive AND
  2. that finding's class is in the management-changing set AND
  3. the Claude layer does not contradict that specific finding.

Everything else REORDERS: the study moves earlier in the worklist by its priority
score but is not paged. Either way the study stays in the queue (invariant 1).

Every decision logs its reason and the exact clause that fired, so the whole split
is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.claude_layer import ClaudeAssessment, _band_from_score, detector_only_assessment
from src.contract import ModuleResult, PanelOutput
from src.logging_conf import get_logger

log = get_logger(__name__)

Action = Literal["page", "reorder"]


@dataclass
class PolicyDecision:
    action: Action
    priority_score: float
    priority_band: str
    reason: str
    paging_findings: list[str] = field(default_factory=list)  # module_ids that justified a page
    audit: list[str] = field(default_factory=list)  # per-clause trace


def _page_candidates(panel: PanelOutput, assessment: ClaudeAssessment) -> tuple[list[ModuleResult], list[str]]:
    """Positive, management-changing, non-contradicted findings + an audit trace."""
    trace: list[str] = []
    candidates: list[ModuleResult] = []
    for r in panel.results:
        if r.status != "ok" or not r.positive:
            continue
        if not r.is_management_changing:
            trace.append(f"{r.module_id}: positive but not management-changing → reorder-only")
            continue
        if assessment.contradicts(r.module_id):
            trace.append(f"{r.module_id}: positive + management-changing but Claude contradicts → not paged")
            continue
        trace.append(f"{r.module_id}: positive + management-changing + not contradicted → PAGE")
        candidates.append(r)
    return candidates, trace


def decide(panel: PanelOutput, assessment: ClaudeAssessment) -> PolicyDecision:
    slog = log.bind(accession=panel.study.accession)
    candidates, trace = _page_candidates(panel, assessment)

    # RANK FLOOR (invariant): Claude may only move a study EARLIER than its
    # detector-only baseline, never later. So the effective reorder priority is
    # floored at the detector-only score — a Claude down-weight can block a page
    # but can never demote a study below where the panel alone would place it.
    baseline = detector_only_assessment(panel).priority_score
    floored = max(assessment.priority_score, baseline)
    if floored > assessment.priority_score:
        trace.append(
            f"rank floor: claude={assessment.priority_score} < detector baseline={baseline} "
            f"→ held at baseline (Claude may only move earlier)"
        )

    if panel.modules_failed:
        trace.append(f"note: modules failed ({', '.join(panel.modules_failed)}) — surfaced, not blocking")

    if candidates:
        decision = PolicyDecision(
            action="page",
            priority_score=max(floored, 0.75),  # a page is at least 'immediate'
            priority_band="immediate",
            reason="page: "
            + "; ".join(f"{r.finding_class.value}={r.value}" for r in candidates),
            paging_findings=[r.module_id for r in candidates],
            audit=trace,
        )
    else:
        band = assessment.priority_band if floored == assessment.priority_score else _band_from_score(floored)
        decision = PolicyDecision(
            action="reorder",
            priority_score=floored,
            priority_band=band,
            reason="reorder: no positive, management-changing, non-contradicted finding",
            paging_findings=[],
            audit=trace,
        )

    slog.info(
        "policy.decision",
        action=decision.action,
        band=decision.priority_band,
        score=decision.priority_score,
        paging=decision.paging_findings,
    )
    return decision
