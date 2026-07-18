"""
Midline-shift module (midline_shift_v1) — approximation.

Wraps the symmetry-based estimator behind the ModuleResult contract:
  value        |shift| in mm
  threshold    5 mm (clinical cutoff)
  detail       signed shift, tilt, ideal/actual midline x, ventricular slice
  approximation=True  (must render visibly in the UI)
"""

from __future__ import annotations

import os

import numpy as np

from src import config
from src.contract import Evidence, ModuleResult
from src.modules.base import Module, timed
from src.modules.midline import estimate_midline_shift
from src.windowing import to_uint8, window_named


def _write_overlay(accession: str, sl: np.ndarray, ideal_x: float, actual_x: float, z: int) -> str:
    from PIL import Image

    out_dir = config.ARTIFACTS_DIR / accession
    out_dir.mkdir(parents=True, exist_ok=True)
    base = to_uint8(window_named(sl, "brain"))
    rgb = np.stack([base, base, base], axis=-1)
    h = rgb.shape[0]
    ix, ax = int(round(ideal_x)), int(round(actual_x))
    if 0 <= ix < rgb.shape[1]:
        rgb[:, ix] = [0, 255, 0]  # ideal midline (green)
    if 0 <= ax < rgb.shape[1]:
        rgb[:, ax] = [255, 0, 0]  # actual deep midline (red)
    path = out_dir / f"midline_{z}.png"
    Image.fromarray(rgb).save(path)
    try:
        return str(path.relative_to(config.REPO_ROOT))
    except ValueError:
        return str(path)


class MidlineShiftModule(Module):
    def __init__(self, module_id: str = "midline_shift_v1"):
        super().__init__(module_id)

    def applies_to(self, ctx) -> bool:
        return self.desc.finding_class.value == "midline_shift" and ctx.modality.upper() == "CT"

    def run(self, volume: np.ndarray, ctx) -> ModuleResult:
        with timed() as t:
            est = estimate_midline_shift(volume, spacing=_spacing_for(ctx))
            overlays = []
            if not os.getenv("TRICORDER_MIDLINE_NO_OVERLAY"):
                overlays.append(
                    _write_overlay(
                        ctx.accession, volume[est.ventricular_slice],
                        est.ideal_midline_x, est.actual_midline_x, est.ventricular_slice,
                    )
                )
            evidence = Evidence(
                slice_indices=[est.ventricular_slice],
                overlay_paths=overlays,
                note=f"symmetry-based proxy; tilt {est.tilt_deg}°, "
                f"signed shift {est.shift_mm_signed} mm",
            )
            return self.build_result(
                value=abs(est.shift_mm_signed),
                detail={
                    "shift_mm_signed": est.shift_mm_signed,
                    "tilt_deg": est.tilt_deg,
                    "ideal_midline_x": est.ideal_midline_x,
                    "actual_midline_x": est.actual_midline_x,
                    "ventricular_slice": float(est.ventricular_slice),
                },
                evidence=evidence,
                runtime_ms=t(),
            )


def _spacing_for(ctx) -> tuple[float, float, float]:
    """Pixel spacing from dicom_meta if present, else 1mm isotropic."""
    try:
        py = float(ctx.dicom_meta.get("PixelSpacingY", "")) if ctx.dicom_meta else 1.0
        px = float(ctx.dicom_meta.get("PixelSpacingX", "")) if ctx.dicom_meta else 1.0
        return (1.0, py, px)
    except (ValueError, TypeError):
        return (1.0, 1.0, 1.0)
