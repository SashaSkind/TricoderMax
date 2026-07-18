"""Orchestrator: unconditional fan-out, failure isolation, superset routing."""

from __future__ import annotations

import numpy as np

from src.contract import StudyContext, StudyType
from src.modules.base import Module
from src.orchestrator import route, run_panel
from src.registry import REGISTRY


def _ctx(**kw) -> StudyContext:
    d = dict(accession="T-1", study_uid="1.2.3", n_slices=16, study_type=StudyType.head_ct_noncontrast)
    d.update(kw)
    return StudyContext(**d)


def _vol():
    return np.random.default_rng(1).normal(30, 20, size=(16, 64, 64)).astype(np.float32)


def test_unknown_study_type_runs_superset():
    _resolved, ids, superset, _reason = route(_ctx(study_type=StudyType.unknown, indication=None))
    assert superset is True
    assert set(ids) == set(REGISTRY.keys())  # invariant 4: full superset


def test_panel_runs_every_selected_module():
    panel = run_panel(_ctx(), _vol())
    assert set(panel.modules_selected) == {r.module_id for r in panel.results}
    assert panel.superset_run is False


def test_failure_isolation(monkeypatch):
    """A crashing module yields status=failed; the pipeline still completes."""
    import src.orchestrator as orch

    real_build = orch.build_module

    def boom_build(mid):
        m = real_build(mid)
        if mid == "ich_v1":
            def crash(volume, ctx):
                raise RuntimeError("boom")
            m.run = crash  # type: ignore[method-assign]
        return m

    monkeypatch.setattr(orch, "build_module", boom_build)
    panel = run_panel(_ctx(study_type=StudyType.unknown), _vol())

    assert "ich_v1" in panel.modules_failed
    ich = next(r for r in panel.results if r.module_id == "ich_v1")
    assert ich.status == "failed" and ich.error
    # Other modules still produced results.
    assert len(panel.results) == len(panel.modules_selected)
    assert len(panel.modules_failed) < len(panel.results)
