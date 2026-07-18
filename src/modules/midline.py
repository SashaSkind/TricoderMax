"""
Midline-shift estimation (approximation).

Head tilt makes the naive vertical image axis a wrong reference, so we estimate
the *expected* midline from the skull's axis of bilateral symmetry (PCA on the
calvarial bone mask → centroid + tilt), NOT from the image axis. We then measure
the horizontal displacement of the deep midline CSF structures (septum pellucidum
/ third ventricle, the dark interhemispheric sliver) from that symmetry line, in
millimetres.

This is a crude proxy — `approximation=True`. The acceptance bar is *directional*
correlation with the CQ500 midline-shift label, not measurement accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MidlineEstimate:
    shift_mm_signed: float  # +/- ; sign = direction of deep-structure displacement
    ventricular_slice: int
    ideal_midline_x: float  # skull symmetry axis (px) at the ventricular row
    actual_midline_x: float  # deep CSF centroid (px)
    tilt_deg: float


def _bone_mask(sl: np.ndarray) -> np.ndarray:
    return sl > 200.0  # calvarium is very dense


def _skull_axis(sl: np.ndarray) -> tuple[float, float, float]:
    """Symmetry axis of the skull via PCA on the bone mask.

    Returns (centroid_x, centroid_y, slope_dx_per_dy). Falls back to the image
    centre if no bone is present.
    """
    ys, xs = np.nonzero(_bone_mask(sl))
    h, w = sl.shape
    if xs.size < 10:
        return w / 2.0, h / 2.0, 0.0
    cx, cy = xs.mean(), ys.mean()
    coords = np.stack([xs - cx, ys - cy], axis=0).astype(np.float64)
    cov = coords @ coords.T / coords.shape[1]
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, int(np.argmax(evals))]  # (dx, dy) of the long axis (≈ vertical)
    dx, dy = major
    if abs(dy) < 1e-6:
        slope = 0.0
    else:
        slope = float(dx / dy)  # x change per unit y along the symmetry axis
    return float(cx), float(cy), slope


def _ventricular_slice(volume: np.ndarray) -> int:
    """Slice in the middle third with the most CSF-density voxels (ventricles)."""
    z = volume.shape[0]
    lo, hi = z // 3, max(z // 3 + 1, 2 * z // 3)
    csf = (volume > 0.0) & (volume < 15.0)
    counts = csf.reshape(z, -1).sum(axis=1)
    band = np.arange(lo, hi)
    return int(band[np.argmax(counts[lo:hi])]) if band.size else z // 2


def _deep_midline_x(sl: np.ndarray, center_x: float) -> float | None:
    """Centroid x of central CSF voxels (interhemispheric midline structures)."""
    h, w = sl.shape
    band = int(w * 0.18)  # central vertical band around the expected midline
    x0, x1 = int(max(0, center_x - band)), int(min(w, center_x + band))
    y0, y1 = int(h * 0.30), int(h * 0.75)  # mid-brain rows (avoid orbit/vertex)
    roi = sl[y0:y1, x0:x1]
    csf = (roi > 0.0) & (roi < 15.0)
    if csf.sum() < 5:
        return None
    xs = np.nonzero(csf)[1]
    return float(x0 + xs.mean())


def estimate_midline_shift(volume: np.ndarray, spacing: tuple[float, float, float]) -> MidlineEstimate:
    z = _ventricular_slice(volume)
    sl = volume[z]
    cx, cy, slope = _skull_axis(sl)
    tilt_deg = float(np.degrees(np.arctan(slope)))

    # ideal midline x at the ventricular row band centre
    ref_row = sl.shape[0] * 0.52
    ideal_x = cx + slope * (ref_row - cy)

    actual_x = _deep_midline_x(sl, ideal_x)
    if actual_x is None:
        actual_x = ideal_x  # no CSF detected → no measurable shift

    spacing_x = spacing[2] if len(spacing) == 3 else 1.0
    shift_mm = (actual_x - ideal_x) * spacing_x
    return MidlineEstimate(
        shift_mm_signed=round(float(shift_mm), 2),
        ventricular_slice=z,
        ideal_midline_x=round(ideal_x, 2),
        actual_midline_x=round(actual_x, 2),
        tilt_deg=round(tilt_deg, 2),
    )
