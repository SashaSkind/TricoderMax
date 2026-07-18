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


def _default_backend() -> str:
    """convnext if its checkpoint is present (strongest), else hf (no download)."""
    from src import config

    override = os.getenv("TRICORDER_ICH_BACKEND", "").strip()
    if override:
        return override
    ckpt = os.getenv("TRICORDER_ICH_CONVNEXT", "ConvNeXt_V2_Model.pth")
    ckpt_path = ckpt if os.path.isabs(ckpt) else str(config.REPO_ROOT / ckpt)
    return "convnext" if os.path.exists(ckpt_path) else "hf"


_BACKEND = _default_backend()


def _get_model():
    global _MODEL
    if _MODEL is None:
        if _BACKEND == "convnext":
            from src.modules.ich_convnext import ConvNeXtICHModel

            _MODEL = ConvNeXtICHModel()
        elif _BACKEND == "hf":
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
        import time

        with timed() as t:
            _t0 = time.time()
            model = _get_model()  # first call loads the checkpoint
            load_ms = int((time.time() - _t0) * 1000)
            three_ch = three_channel(volume)  # (Z,3,Y,X)

            _t1 = time.time()
            per_slice_any, subtypes = model.score(three_ch)
            infer_ms = int((time.time() - _t1) * 1000)
            n = int(three_ch.shape[0])
            timing = f"load={load_ms}ms inference={infer_ms}ms ({infer_ms/max(1,n):.0f}ms/slice, {n} slices)"
            study_any = float(np.quantile(per_slice_any, 0.98)) if per_slice_any.size else 0.0
            top_slices = np.argsort(-per_slice_any)[:3].astype(int).tolist()

            overlays: list[str] = []
            note_bits = [f"backend={_BACKEND}", timing]
            if getattr(model, "supports_gradcam", False) and not os.getenv("TRICORDER_ICH_NO_OVERLAY"):
                # A heatmap failure must never sink the detection — keep the score.
                try:
                    z0 = top_slices[0]
                    cam = model.gradcam(three_ch[z0])
                    overlays.append(_write_overlay(ctx.accession, z0, window_named(volume[z0], "brain"), cam))
                except Exception as e:  # noqa: BLE001
                    note_bits.append(f"heatmap unavailable ({type(e).__name__})")
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
