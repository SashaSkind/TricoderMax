"""
Phase 2 acceptance (wiring): the ICH module returns a schema-valid result with
populated evidence.slice_indices and at least one overlay image on disk.

Uses an UNTRAINED model (no real weights available in CI) — this validates the
plumbing and Grad-CAM path, not detection accuracy. Real weights: see
scripts/fetch_ich_weights.md.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# Keep the wiring test small/fast, force the local timm backend + untrained path.
os.environ["TRICORDER_ICH_BACKEND"] = "timm"
os.environ.setdefault("TRICORDER_ICH_BACKBONE", "tf_efficientnet_b0")
os.environ.setdefault("TRICORDER_ICH_INPUT", "96")
os.environ["TRICORDER_ICH_ALLOW_RANDOM"] = "1"

pytest.importorskip("torch")
pytest.importorskip("timm")

from src.contract import ModuleResult, StudyContext, StudyType  # noqa: E402


def _ctx():
    return StudyContext(
        accession="ICH-TEST-1", study_uid="1.2.3",
        study_type=StudyType.head_ct_noncontrast, modality="CT", n_slices=8,
    )


def _synthetic_ich_volume():
    """(Z,Y,X) HU volume with a bright focal 'bleed' on a couple of slices."""
    rng = np.random.default_rng(0)
    vol = rng.normal(30, 10, size=(8, 64, 64)).astype(np.float32)
    vol[3:5, 28:36, 28:36] += 45.0  # focal hyperdensity
    # calvarial ring
    vol[:, 0, :] = vol[:, -1, :] = vol[:, :, 0] = vol[:, :, -1] = 900.0
    return vol


def test_ich_module_contract_and_overlay(tmp_path, monkeypatch):
    from src import config
    from src.modules import ich as ich_mod
    from src.modules.ich import ICHModule

    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    ich_mod._MODEL = None  # reset process cache

    module = ICHModule()
    result = module.run(_synthetic_ich_volume(), _ctx())

    assert isinstance(result, ModuleResult)
    assert result.status == "ok"
    assert result.finding_class.value == "intracranial_hemorrhage"
    assert 0.0 <= result.value <= 1.0
    # per-subtype detail present
    assert set(result.detail) == {"epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"}
    # evidence: slice indices + at least one overlay on disk
    assert result.evidence.slice_indices, "expected top slice indices"
    assert result.evidence.overlay_paths, "expected at least one overlay path"
    from pathlib import Path

    overlay_abs = Path(result.evidence.overlay_paths[0])
    if not overlay_abs.is_absolute():
        overlay_abs = config.REPO_ROOT / overlay_abs
    assert overlay_abs.exists() and overlay_abs.stat().st_size > 0
    # untrained → must be flagged uncalibrated and noted
    assert result.calibration is None
    assert "UNTRAINED" in (result.evidence.note or "")


def test_ich_aggregate_and_gradcam_shapes():
    from src.modules.ich_model import ICHModel, aggregate
    from src.windowing import three_channel

    ich_mod = ICHModel()
    vol = _synthetic_ich_volume()
    scores = ich_mod.score_volume(three_channel(vol))
    assert scores.per_slice.shape == (8, 6)
    agg = aggregate(scores)
    assert 0.0 <= agg["study_any"] <= 1.0
    cam = ich_mod.gradcam(three_channel(vol)[3])
    assert cam.shape == vol.shape[1:]
    assert 0.0 <= float(cam.min()) and float(cam.max()) <= 1.0 + 1e-6
