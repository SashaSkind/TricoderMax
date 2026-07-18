"""
Phase 5: Claude layer structured output + fallback, and the acceptance scenario
(a high detector probability correctly down-weighted by clinical context).

The real Anthropic call is exercised via an injected fake client, so tests are
deterministic and need no API key or network.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from src.claude_layer import assess, detector_only_assessment
from src.contract import FindingClass, ModuleResult, PanelOutput, StudyContext
from src.policy import decide


def _ich(value: float) -> ModuleResult:
    return ModuleResult(
        module_id="ich_v1", module_version="0.1.0",
        finding_class=FindingClass.intracranial_hemorrhage,
        result_type="probability", value=value, threshold=0.55,
    )


def _panel(results, **study_kw) -> PanelOutput:
    ctx = StudyContext(accession="C-1", study_uid="1", n_slices=20, **study_kw)
    now = datetime.now(timezone.utc)
    return PanelOutput(
        study=ctx, results=results, modules_selected=[r.module_id for r in results],
        modules_failed=[r.module_id for r in results if r.status == "failed"],
        started_at=now, completed_at=now, total_runtime_ms=1,
    )


class _FakeClient:
    """Mimics anthropic.Anthropic: returns a forced tool_use with `payload`."""

    def __init__(self, payload: dict):
        block = SimpleNamespace(type="tool_use", name="record_assessment", input=payload)
        resp = SimpleNamespace(content=[block])
        self.messages = SimpleNamespace(create=lambda **kw: resp)


class _ScriptedClient:
    """Returns a scripted sequence of responses across successive create() calls."""

    def __init__(self, script):
        self.script, self.i = script, 0
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        r = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return SimpleNamespace(content=r)


def _tooluse(name, inp, tid="t1"):
    return SimpleNamespace(type="tool_use", name=name, input=inp, id=tid)


def test_hot_path_tool_loop_executes_then_records():
    """Claude requests a bone-window slice, then records — the loop runs the tool
    and returns the parsed assessment."""
    import numpy as np

    from src.contract import StudyContext

    panel = _panel([_ich(0.9)])
    ctx = StudyContext(accession="C-1", study_uid="1", n_slices=16)
    vol = np.random.default_rng(0).normal(30, 20, size=(16, 64, 64)).astype(np.float32)
    payload = {
        "priority_score": 0.8, "priority_band": "immediate",
        "verification": [{"module_id": "ich_v1", "supported": True, "reasoning": "confirmed on bone window", "confidence_adjustment": 0.0}],
        "summary": "Acute ICH.", "caveats": [], "abstain": False,
    }
    script = [
        [_tooluse("get_slices", {"start": 8, "end": 8, "window": "bone"})],  # request evidence
        [_tooluse("record_assessment", payload)],                            # then decide
    ]
    a = assess(panel, client=_ScriptedClient(script), volume=vol, ctx=ctx)
    assert a.source == "claude" and a.priority_band == "immediate"


def test_no_client_uses_detector_only_fallback():
    panel = _panel([_ich(0.9)])
    a = assess(panel, client=None)
    assert a.source == "fallback"


def test_structured_response_parsed():
    panel = _panel([_ich(0.9)])
    payload = {
        "priority_score": 0.82, "priority_band": "immediate",
        "verification": [{"module_id": "ich_v1", "supported": True, "reasoning": "clear bleed"}],
        "summary": "Acute ICH, page.", "caveats": [], "abstain": False,
    }
    a = assess(panel, client=_FakeClient(payload))
    assert a.source == "claude" and a.priority_band == "immediate"
    assert a.verification[0].supported is True


def test_abstain_falls_back():
    panel = _panel([_ich(0.9)])
    payload = {
        "priority_score": 0.0, "priority_band": "routine",
        "verification": [], "summary": "unsure", "caveats": [], "abstain": True,
    }
    a = assess(panel, client=_FakeClient(payload))
    assert a.source == "fallback"
    assert any("abstain" in c.lower() for c in a.caveats)


def test_malformed_response_falls_back():
    panel = _panel([_ich(0.9)])
    bad = {"priority_score": 5.0, "priority_band": "nope"}  # out of range + bad enum
    a = assess(panel, client=_FakeClient(bad))
    assert a.source == "fallback"


def test_acceptance_high_prob_downweighted_by_context_reorders():
    """High ICH prob + Claude says unsupported (partial-volume, young, no trauma)
    → policy must NOT page; the study reorders but stays in the queue."""
    panel = _panel(
        [_ich(0.88)], age=24, indication="Headache, no trauma. Screening."
    )
    payload = {
        "priority_score": 0.18, "priority_band": "routine",
        "verification": [
            {
                "module_id": "ich_v1", "supported": False,
                "reasoning": "heatmap over skull-base partial-volume; young pt, no trauma history",
            }
        ],
        "summary": "Likely partial-volume artifact, not acute ICH.",
        "caveats": ["Down-weighted high subdural score by context."], "abstain": False,
    }
    a = assess(panel, client=_FakeClient(payload))
    assert a.source == "claude"
    assert a.contradicts("ich_v1") is True

    decision = decide(panel, a)
    assert decision.action == "reorder"  # NOT paged, but still in the queue
    assert any("contradicts" in line for line in decision.audit)

    # Contrast: the detector-only fallback WOULD have paged the same panel.
    fb_decision = decide(panel, detector_only_assessment(panel))
    assert fb_decision.action == "page"
