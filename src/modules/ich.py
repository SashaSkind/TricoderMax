"""
Intracranial-hemorrhage module (ich_v1).

Wraps a reused open-weight model behind the ModuleResult contract:
  volume → 3-channel windows → per-slice ICH probs → study aggregate (+ Grad-CAM
  overlays when the backend supports them).

Backends (TRICORDER_ICH_BACKEND):
  "hf"   → Hugging Face RSNA-2019 ViT (real open weights, no Kaggle; weak). Default.
  "timm" → local RSNA-2019 CNN checkpoint at weights/ich/ich_cnn.pth (see
           scripts/fetch_ich_weights.py); Grad-CAM overlays supported.

Runs strictly as a detector (no LLM, no context). Emits value=P(any ICH),
detail=subtypes (when available), evidence=top slices (+overlays).
"""

from __future__ import annotations

import os

import numpy as np

from src import config
from src.contract import Evidence, ModuleResult
from src.modules.base import Module, timed
from src.windowing import three_channel, to_uint8, window_named

_MODEL = None  # process-wide cache; the checkpoint loads once
_BACKEND = os.getenv("TRICORDER_ICH_BACKEND", "hf")


def _get_model():
    global _MODEL
    if _MODEL is None:
        if _BACKEND == "hf":
            from src.modules.ich_hf import HFICHModel

            _MODEL = HFICHModel()
        else:
            from src.modules.ich_model import ICHModel

            _MODEL = ICHModel()
    return _MODEL


def _write_overlay(accession: str, z: int, brain_slice: np.ndarray, cam: np.ndarray) -> str:
    from PIL import Image

    out_dir = config.ARTIFACTS_DIR / accession
    out_dir.mkdir(parents=True, exist_ok=True)
    base = to_uint8(brain_slice)
    rgb = np.stack([base, base, base], axis=-1).astype(np.float32)
    heat = np.zeros_like(rgb)
    heat[..., 0] = to_uint8(cam)
    blend = (0.6 * rgb + 0.4 * heat).clip(0, 255).astype(np.uint8)
    path = out_dir / f"cam_{z}.png"
    Image.fromarray(blend).save(path)
    try:
        return str(path.relative_to(config.REPO_ROOT))
    except ValueError:
        return str(path)


class ICHModule(Module):
    def __init__(self, module_id: str = "ich_v1"):
        super().__init__(module_id)

    def applies_to(self, ctx) -> bool:
        return self.desc.finding_class.value == "intracranial_hemorrhage" and ctx.modality.upper() == "CT"

    def run(self, volume: np.ndarray, ctx) -> ModuleResult:
        with timed() as t:
            model = _get_model()
            three_ch = three_channel(volume)  # (Z,3,Y,X)

            per_slice_any, subtypes = model.score(three_ch)
            study_any = float(np.quantile(per_slice_any, 0.98)) if per_slice_any.size else 0.0
            top_slices = np.argsort(-per_slice_any)[:3].astype(int).tolist()

            overlays: list[str] = []
            note_bits = [f"backend={_BACKEND}"]
            if getattr(model, "supports_gradcam", False) and not os.getenv("TRICORDER_ICH_NO_OVERLAY"):
                cam = model.gradcam(three_ch[top_slices[0]])
                overlays.append(_write_overlay(ctx.accession, top_slices[0], window_named(volume[top_slices[0]], "brain"), cam))
            if getattr(model, "untrained", False):
                note_bits.append("UNTRAINED MODEL — wiring test only, not real detection")

            evidence = Evidence(
                slice_indices=top_slices,
                overlay_paths=overlays,
                note="; ".join(note_bits),
            )
            result = self.build_result(
                value=round(study_any, 4),
                detail=subtypes,
                evidence=evidence,
                runtime_ms=t(),
            )
            # Record which real model produced this (provenance in logs/audit).
            result.evidence.note += f"; model={getattr(model, 'model_name', 'ich')}"
            if getattr(model, "untrained", False):
                result.calibration = None
            return result
