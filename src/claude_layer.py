"""
Claude layer — runs strictly DOWNSTREAM of the detector panel (invariant 2).

It aggregates all ModuleResults + clinical context, verifies each detector claim
against that context, and produces a priority ranking. It NEVER decides which
detectors run and NEVER removes a study from the queue — the worst it can do is
decline to escalate, which still leaves the study in the reading worklist.

Structured output is enforced via a forced tool call whose input_schema is
derived from `ClaudeAssessment`. On ANY failure (no API key, network/parse error,
schema-invalid, or `abstain: true`) we fall back to the deterministic
detector-only ranking — never to dropping the study.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src import config
from src.contract import PanelOutput
from src.logging_conf import get_logger

log = get_logger(__name__)

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


# ── deterministic fallback ───────────────────────────────────────────────────
def detector_only_assessment(panel: PanelOutput) -> ClaudeAssessment:
    """Rank from detector output alone. Every positive is 'supported' (we have no
    context to contradict it — the conservative choice: never suppress)."""
    positives = panel.positive_results
    mgmt_positive = [r for r in positives if r.is_management_changing]

    def signal(r) -> float:
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
            module_id=r.module_id, supported=True,
            reasoning="detector-only fallback: no clinical context available to verify",
        )
        for r in positives
    ]
    caveats = ["Detector-only ranking (Claude layer not invoked)."]
    if panel.modules_failed:
        caveats.append(f"Modules failed and were not considered: {', '.join(panel.modules_failed)}.")
    labels = ", ".join(sorted({r.finding_class.value for r in positives})) or "no positive findings"
    return ClaudeAssessment(
        priority_score=round(score, 3), priority_band=_band_from_score(score),
        verification=verification, summary=f"Detector panel flagged: {labels}.",
        caveats=caveats, abstain=False, source="fallback",
    )


# ── prompt assembly ──────────────────────────────────────────────────────────
_SYSTEM = (
    "You are the reconciliation layer of a head-CT triage system. You run strictly "
    "DOWNSTREAM of a fixed detector panel. Your ONLY job is to PRIORITIZE studies for "
    "a radiologist who will read every study regardless. You never remove a study from "
    "the queue and you never choose which detectors run.\n\n"
    "For each detector finding, judge whether the clinical context (age, indication, "
    "timing, prior report) and the provided candidate slice images SUPPORT or fail to "
    "support it — e.g. a high subdural probability whose heatmap sits over a skull-base "
    "partial-volume region in a young patient with no trauma history should be marked "
    "unsupported. Down-weight unsupported findings; never fabricate findings the panel "
    "did not report. If you are not confident, set abstain=true and the system will use "
    "the detector-only ranking. Report the EXTERNAL calibration you are given; do not "
    "invent numbers."
)


def _serialize_results(panel: PanelOutput) -> list[dict]:
    out = []
    for r in panel.results:
        out.append(
            {
                "module_id": r.module_id, "finding_class": r.finding_class.value,
                "status": r.status, "result_type": r.result_type, "value": r.value,
                "units": r.units, "threshold": r.threshold, "positive": r.positive,
                "approximation": r.approximation, "detail": r.detail,
                "management_changing": r.is_management_changing,
                "calibration": (r.calibration.model_dump() if r.calibration else None),
                "evidence_slices": r.evidence.slice_indices, "error": r.error,
            }
        )
    return out


def _image_blocks(panel: PanelOutput, max_images: int = 4) -> list[dict]:
    blocks: list[dict] = []
    for r in panel.results:
        for rel in r.evidence.overlay_paths:
            p = Path(rel)
            if not p.is_absolute():
                p = config.REPO_ROOT / p
            if p.exists() and len(blocks) < max_images:
                data = base64.standard_b64encode(p.read_bytes()).decode()
                blocks.append(
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}
                )
    return blocks


def _tool_schema() -> dict:
    schema = ClaudeAssessment.model_json_schema()
    # 'source' is ours, not the model's, and $defs must be inlined for the tool.
    props = schema.get("properties", {})
    props.pop("source", None)
    if "required" in schema:
        schema["required"] = [f for f in schema["required"] if f != "source"]
    return schema


def _build_messages(panel: PanelOutput, prior_report: Optional[str]) -> list[dict]:
    s = panel.study
    context = {
        "accession": s.accession, "study_type": s.study_type.value,
        "age": s.age, "sex": s.sex, "indication": s.indication,
        "acquired_at": str(s.acquired_at) if s.acquired_at else None,
        "prior_report": prior_report or s.prior_report,
        "routing_reason": panel.routing_reason,
        "modules_failed": panel.modules_failed,
    }
    text = (
        "STUDY CONTEXT:\n" + json.dumps(context, indent=2, default=str)
        + "\n\nDETECTOR PANEL RESULTS:\n" + json.dumps(_serialize_results(panel), indent=2, default=str)
        + "\n\nAttached images are the detectors' top candidate slices (overlays). "
        "Return your assessment via the record_assessment tool."
    )
    content: list[dict] = [{"type": "text", "text": text}] + _image_blocks(panel)
    return [{"role": "user", "content": content}]


def _default_client():
    key = config.ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        return None
    try:
        import anthropic
    except ImportError:
        log.warning("claude.no_sdk", hint="pip install '.[llm]'")
        return None
    return anthropic.Anthropic(api_key=key)


def _call_claude(client, panel: PanelOutput, prior_report: Optional[str]) -> dict:
    tool = {
        "name": "record_assessment",
        "description": "Record the triage priority assessment.",
        "input_schema": _tool_schema(),
    }
    resp = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=1024,
        system=_SYSTEM,
        tools=[tool],
        tool_choice={"type": "tool", "name": "record_assessment"},
        messages=_build_messages(panel, prior_report),
    )
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "record_assessment":
            return dict(block.input)
    raise ValueError("no record_assessment tool_use block in response")


# ── entry point ──────────────────────────────────────────────────────────────
def assess(panel: PanelOutput, prior_report: Optional[str] = None, client: Any = "auto") -> ClaudeAssessment:
    """Reconcile + rank. Falls back to detector-only on any failure or abstain."""
    if client == "auto":
        client = _default_client()
    if client is None:
        return detector_only_assessment(panel)

    slog = log.bind(accession=panel.study.accession)
    try:
        raw = _call_claude(client, panel, prior_report)
        assessment = ClaudeAssessment(**raw, source="claude")
    except (ValidationError, ValueError, KeyError) as e:
        slog.warning("claude.parse_failed", err=str(e))
        return detector_only_assessment(panel)
    except Exception as e:  # noqa: BLE001 — network/SDK errors → safe fallback
        slog.warning("claude.call_failed", err=f"{type(e).__name__}: {e}")
        return detector_only_assessment(panel)

    if assessment.abstain:
        slog.info("claude.abstain", detail="using detector-only ranking")
        fb = detector_only_assessment(panel)
        fb.caveats.append("Claude abstained; detector-only ranking used.")
        return fb
    slog.info("claude.ok", band=assessment.priority_band, score=assessment.priority_score)
    return assessment
