"""
Triage pipeline — the end-to-end path for one study, and worklist ranking.

    study + volume
      -> run_panel        (detector fan-out)
      -> assess           (Claude layer, downstream only; fallback in Phase 0)
      -> decide           (deterministic policy)
      = TriageResult

The worklist is then ordered by (paged first, then priority score). Crucially,
ordering only REORDERS — nothing is ever dropped (invariant 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src import config
from src.claude_layer import ClaudeAssessment, assess
from src.contract import PanelOutput, StudyContext
from src.orchestrator import run_panel
from src.policy import PolicyDecision, decide


@dataclass
class TriageResult:
    panel: PanelOutput
    assessment: ClaudeAssessment
    decision: PolicyDecision

    @property
    def study(self) -> StudyContext:
        return self.panel.study

    @property
    def sort_key(self) -> tuple:
        # Paged studies first; then by priority score desc; stable by accession.
        return (0 if self.decision.action == "page" else 1, -self.decision.priority_score, self.study.accession)


def triage_study(ctx: StudyContext, volume: np.ndarray) -> TriageResult:
    panel = run_panel(ctx, volume)
    # Mock panels never go to the real Claude API (pointless + costs money); they
    # use the deterministic fallback. Real detector runs (USE_MOCKS=0) invoke Claude.
    client = None if config.USE_MOCKS else "auto"
    assessment = assess(panel, prior_report=ctx.prior_report, client=client)
    decision = decide(panel, assessment)
    return TriageResult(panel=panel, assessment=assessment, decision=decision)


def rank_worklist(results: list[TriageResult]) -> list[TriageResult]:
    """Reorder only — every input study is present in the output."""
    return sorted(results, key=lambda r: r.sort_key)


def triage_real_study(series_dir: str) -> TriageResult:
    """Run the REAL detector panel (not mocks) on a DICOM series directory.

    Instantiates the real ICH + midline modules directly (bypassing the mock-gated
    factory), then runs the Claude layer (if ANTHROPIC_API_KEY is set) and policy.
    Used by run_study.py and the UI's real-study view.
    """
    from datetime import datetime, timezone

    from src.contract import PanelOutput
    from src.dicom_io import load_series, read_study_context
    from src.modules.ich import ICHModule
    from src.modules.midline_shift import MidlineShiftModule

    vol = load_series(series_dir)
    ctx = read_study_context(series_dir)
    modules = [ICHModule(), MidlineShiftModule()]

    started = datetime.now(timezone.utc)
    results = []
    for m in modules:
        try:
            results.append(m.run(vol.hu, ctx) if m.applies_to(ctx) else m.not_applicable_result())
        except Exception as e:  # noqa: BLE001 — failure isolation
            results.append(m.failed_result(f"{type(e).__name__}: {e}"))
    completed = datetime.now(timezone.utc)

    panel = PanelOutput(
        study=ctx,
        results=results,
        modules_selected=[m.module_id for m in modules],
        modules_failed=[r.module_id for r in results if r.status == "failed"],
        started_at=started,
        completed_at=completed,
        total_runtime_ms=int((completed - started).total_seconds() * 1000),
    )
    assessment = assess(panel, prior_report=ctx.prior_report, client="auto")
    decision = decide(panel, assessment)
    return TriageResult(panel=panel, assessment=assessment, decision=decision)
