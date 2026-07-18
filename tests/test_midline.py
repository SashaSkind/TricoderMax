"""
Phase 3: midline-shift approximation.

Uses controlled phantoms (a symmetric skull with a deep midline structure
displaced by a KNOWN amount) to check the estimator recovers the correct
direction and approximate magnitude — the directional-correlation bar, applied to
synthetic ground truth since CQ500 is not present.
"""

from __future__ import annotations

import numpy as np

from src.contract import StudyContext, StudyType
from src.modules.midline import estimate_midline_shift
from src.modules.midline_shift import MidlineShiftModule


def _phantom(shift_px: int, spacing_x: float = 0.5, z=12, size=128) -> np.ndarray:
    """Symmetric elliptical skull + brain; deep CSF blob displaced by shift_px."""
    vol = np.full((z, size, size), 30.0, dtype=np.float32)  # brain ~30 HU
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = size // 2
    # taller-than-wide ellipse ring → symmetry axis is vertical
    r = ((xx - cx) ** 2) / (50.0**2) + ((yy - cy) ** 2) / (60.0**2)
    ring = (r > 0.9) & (r < 1.05)
    outside = r >= 1.05
    for k in range(z):
        sl = vol[k]
        sl[ring] = 900.0  # calvarium
        sl[outside] = -1000.0  # air outside the skull
    # deep midline CSF structure on the mid slices, displaced by shift_px in x
    for k in range(z // 3, 2 * z // 3):
        vol[k, cy - 6 : cy + 6, cx + shift_px - 3 : cx + shift_px + 3] = 5.0
    return vol


def _ctx(spacing_x=0.5):
    return StudyContext(
        accession="MLS-1", study_uid="1", study_type=StudyType.head_ct_noncontrast,
        modality="CT", n_slices=12,
        dicom_meta={"PixelSpacingX": str(spacing_x), "PixelSpacingY": str(spacing_x)},
    )


def test_direction_positive_shift():
    vol = _phantom(shift_px=16, spacing_x=0.5)
    est = estimate_midline_shift(vol, spacing=(1.0, 0.5, 0.5))
    assert est.shift_mm_signed > 0  # displaced to +x
    assert est.shift_mm_signed > 3  # ~16px * 0.5mm ≈ 8mm, robustly > threshold-ish


def test_direction_negative_shift():
    vol = _phantom(shift_px=-16, spacing_x=0.5)
    est = estimate_midline_shift(vol, spacing=(1.0, 0.5, 0.5))
    assert est.shift_mm_signed < 0


def test_no_shift_is_small():
    vol = _phantom(shift_px=0)
    est = estimate_midline_shift(vol, spacing=(1.0, 0.5, 0.5))
    assert abs(est.shift_mm_signed) < 2.0


def test_directional_monotonic():
    """More displacement → larger measured shift (directional correlation)."""
    spacing = (1.0, 0.5, 0.5)
    vals = [estimate_midline_shift(_phantom(px), spacing).shift_mm_signed for px in (0, 8, 16, 24)]
    assert vals[0] < vals[1] < vals[2] < vals[3]


def test_module_contract(tmp_path, monkeypatch):
    from src import config

    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    m = MidlineShiftModule()
    res = m.run(_phantom(shift_px=16), _ctx())
    assert res.status == "ok"
    assert res.result_type == "measurement" and res.units == "mm"
    assert res.approximation is True  # must render visibly
    assert res.value == abs(res.detail["shift_mm_signed"])
    assert res.threshold == 5.0
    assert res.evidence.slice_indices
    assert res.evidence.overlay_paths  # midline overlay written
