"""
Intracranial-hemorrhage module (ich_v1).

Wraps a reused pretrained CNN behind the ModuleResult contract:
  volume → 3-channel windows → per-slice ICH probs → study aggregate → Grad-CAM
  overlays on the top slices.

Runs strictly as a detector (no LLM, no context). Emits:
  value            study-level P(any ICH)
  detail           per-subtype probabilities
  evidence         top slice indices + overlay PNG paths (under artifacts/)
"""

from __future__ import annotations

import os

import numpy as np

from src import config
from src.contract import Evidence, ModuleResult
from src.modules.base import Module, timed
from src.windowing import three_channel, to_uint8, window_named

_MODEL = None  # process-wide cache; the checkpoint loads once


def _get_model():
    global _MODEL
    if _MODEL is None:
        from src.modules.ich_model import ICHModel

        _MODEL = ICHModel()
    return _MODEL


def _write_overlay(accession: str, z: int, brain_slice: np.ndarray, cam: np.ndarray) -> str:
    from PIL import Image

    out_dir = config.ARTIFACTS_DIR / accession
    out_dir.mkdir(parents=True, exist_ok=True)
    base = to_uint8(brain_slice)  # (Y,X) grayscale
    rgb = np.stack([base, base, base], axis=-1).astype(np.float32)
    heat = np.zeros_like(rgb)
    heat[..., 0] = to_uint8(cam)  # red channel = CAM
    blend = (0.6 * rgb + 0.4 * heat).clip(0, 255).astype(np.uint8)
    path = out_dir / f"cam_{z}.png"
    Image.fromarray(blend).save(path)
    # Store a repo-relative path in the contract.
    try:
        return str(path.relative_to(config.REPO_ROOT))
    except ValueError:
        return str(path)


class ICHModule(Module):
    def __init__(self, module_id: str = "ich_v1"):
        super().__init__(module_id)

    def applies_to(self, ctx) -> bool:
        # Metadata only: any head CT (or ambiguous). Never inspect pixels here.
        return self.desc.finding_class.value == "intracranial_hemorrhage" and ctx.modality.upper() == "CT"

    def run(self, volume: np.ndarray, ctx) -> ModuleResult:
        with timed() as t:
            model = _get_model()
            three_ch = three_channel(volume)  # (Z,3,Y,X)
            from src.modules.ich_model import aggregate

            scores = model.score_volume(three_ch)
            agg = aggregate(scores)

            overlays: list[str] = []
            note_bits = []
            if not os.getenv("TRICORDER_ICH_NO_OVERLAY"):
                for z in agg["top_slices"][:1]:  # overlay the single top slice
                    cam = model.gradcam(three_ch[z])
                    brain = window_named(volume[z], "brain")
                    overlays.append(_write_overlay(ctx.accession, z, brain, cam))

            if getattr(model, "untrained", False):
                note_bits.append(
                    "UNTRAINED MODEL (TRICORDER_ICH_ALLOW_RANDOM) — wiring test only, not real detection"
                )

            evidence = Evidence(
                slice_indices=agg["top_slices"],
                overlay_paths=overlays,
                note="; ".join(note_bits) or None,
            )
            # Untrained → force uncalibrated (never attach the registry's calibration).
            approx = self.desc.approximation
            result = self.build_result(
                value=agg["study_any"],
                detail=agg["subtypes"],
                evidence=evidence,
                runtime_ms=t(),
                approximation=approx,
            )
            if getattr(model, "untrained", False):
                result.calibration = None
            return result
