"""
ICH model wrapper + Grad-CAM. Torch-only; imported lazily by the module.

Reference architecture: a 2D CNN classifier over per-slice 3-channel
(brain/subdural/bone) windows, emitting the RSNA-2019 label set
(5 subtypes + "any"). This matches the wrappable public checkpoints (e.g. the
VinBigData RSNA-2019 CNN stage); a sequence head can be added later without
changing this module's contract.

Weights: reused, not trained (see scripts/fetch_ich_weights.md). If no checkpoint
is present, set TRICORDER_ICH_ALLOW_RANDOM=1 to build an UNTRAINED model for
wiring/plumbing tests only — results are marked uncalibrated with a loud note and
must never be reported as real detection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# RSNA-2019 ICH label set. "any" is the study-level positive signal.
SUBTYPES = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
LABELS = SUBTYPES + ["any"]
DEFAULT_BACKBONE = os.getenv("TRICORDER_ICH_BACKBONE", "tf_efficientnet_b0")
INPUT_HW = int(os.getenv("TRICORDER_ICH_INPUT", "384"))


@dataclass
class SliceScores:
    per_slice: np.ndarray  # (Z, 6) sigmoid probs, columns = LABELS
    input_hw: int


def _manifest_backbone() -> str:
    """Backbone recorded by scripts/fetch_ich_weights.py, if a manifest exists."""
    from src import config

    mf = config.WEIGHTS_DIR / "ich" / "manifest.json"
    if mf.exists():
        try:
            import json

            return json.loads(mf.read_text()).get("backbone", DEFAULT_BACKBONE)
        except Exception:  # noqa: BLE001
            pass
    return DEFAULT_BACKBONE


def build_model(num_classes: int = len(LABELS), backbone: str | None = None):
    import timm

    return timm.create_model(
        backbone or _manifest_backbone(), pretrained=False, num_classes=num_classes, in_chans=3
    )


def _resolve_weights(weights_path: str | None):
    from pathlib import Path

    from src import config

    if weights_path and Path(weights_path).exists():
        return Path(weights_path)
    default = config.WEIGHTS_DIR / "ich" / "ich_cnn.pth"
    return default if default.exists() else None


class ICHModel:
    """Loads the checkpoint (or an untrained model in dev) and scores a volume."""

    def __init__(self, weights_path: str | None = None):
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = _resolve_weights(weights_path)
        self.model = build_model().to(self.device).eval()
        self.untrained = False

        if ckpt is not None:
            state = torch.load(ckpt, map_location=self.device)
            for k in ("state_dict", "model", "model_state_dict"):
                if isinstance(state, dict) and k in state:
                    state = state[k]
                    break
            state = {kk.replace("module.", ""): vv for kk, vv in state.items()}
            self.model.load_state_dict(state, strict=False)
        elif os.getenv("TRICORDER_ICH_ALLOW_RANDOM", "").strip() in ("1", "true", "True"):
            self.untrained = True  # wiring-test model — NOT real detection
        else:
            raise FileNotFoundError(
                "No ICH checkpoint found. Place weights at weights/ich/ich_cnn.pth "
                "(see scripts/fetch_ich_weights.md) or set TRICORDER_ICH_ALLOW_RANDOM=1 "
                "for an untrained wiring test."
            )

    # ── inference ────────────────────────────────────────────────────────────
    def score_volume(self, three_ch: np.ndarray, batch: int = 16) -> SliceScores:
        """three_ch: (Z, 3, Y, X) float32 in [0,1] → per-slice (Z, 6) probs."""
        import torch
        import torch.nn.functional as F

        z = three_ch.shape[0]
        out = np.zeros((z, len(LABELS)), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, z, batch):
                chunk = torch.from_numpy(three_ch[i : i + batch]).float().to(self.device)
                chunk = F.interpolate(chunk, size=(INPUT_HW, INPUT_HW), mode="bilinear", align_corners=False)
                logits = self.model(chunk)
                out[i : i + batch] = torch.sigmoid(logits).cpu().numpy()
        return SliceScores(per_slice=out, input_hw=INPUT_HW)

    # ── Grad-CAM for the "any" logit on one slice ──────────────────────────────
    def gradcam(self, slice_3ch: np.ndarray) -> np.ndarray:
        """slice_3ch: (3, Y, X) → CAM heatmap (Y, X) in [0,1]."""
        import torch
        import torch.nn.functional as F

        feats: dict = {}
        target_layer = self._last_conv()

        def fwd_hook(_m, _i, o):
            feats["act"] = o

        def bwd_hook(_m, _gi, go):
            feats["grad"] = go[0]

        h1 = target_layer.register_forward_hook(fwd_hook)
        h2 = target_layer.register_full_backward_hook(bwd_hook)
        try:
            x = torch.from_numpy(slice_3ch[None]).float().to(self.device)
            x = F.interpolate(x, size=(INPUT_HW, INPUT_HW), mode="bilinear", align_corners=False)
            x.requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x)
            logits[0, LABELS.index("any")].backward()
            act = feats["act"][0]  # (C, h, w)
            grad = feats["grad"][0]  # (C, h, w)
            weights = grad.mean(dim=(1, 2), keepdim=True)
            cam = torch.relu((weights * act).sum(0))
            cam = cam / (cam.max() + 1e-6)
            cam = F.interpolate(cam[None, None], size=slice_3ch.shape[1:], mode="bilinear", align_corners=False)
            return cam[0, 0].detach().cpu().numpy()
        finally:
            h1.remove()
            h2.remove()

    def _last_conv(self):
        import torch.nn as nn

        last = None
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                last = m
        if last is None:
            raise RuntimeError("no Conv2d layer found for Grad-CAM")
        return last


def aggregate(scores: SliceScores) -> dict:
    """Per-slice probs → study-level summary.

    Study 'any' = high-quantile of per-slice 'any' (robust to a single hot slice
    while still firing on focal bleeds). Subtypes = max over slices.
    """
    ps = scores.per_slice
    any_col = ps[:, LABELS.index("any")]
    study_any = float(np.quantile(any_col, 0.98)) if len(any_col) else 0.0
    subtypes = {s: round(float(ps[:, LABELS.index(s)].max()), 4) for s in SUBTYPES}
    top_slices = np.argsort(-any_col)[:3].tolist()
    return {
        "study_any": round(study_any, 4),
        "subtypes": subtypes,
        "top_slices": [int(i) for i in top_slices],
        "per_slice_any": any_col,
    }
