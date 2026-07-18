"""
ConvNeXt V2 ICH backend — a REAL, strong open-weight RSNA-2019 model.

Loads the Kaggle "ConvNeXt V2 Model" checkpoint (Apache-2.0): a timm
`convnextv2_base` backbone + a custom head
    Linear(1024→512) → BatchNorm1d(512) → ReLU → Dropout → Linear(512→5)
emitting 5 ICH outputs. Replicated exactly here so the checkpoint loads ~100%.

Place the file at repo root as `ConvNeXt_V2_Model.pth` (default) or set
TRICORDER_ICH_CONVNEXT to its path, then select this backend with
TRICORDER_ICH_BACKEND=convnext.

CAVEAT: the checkpoint loads exactly, but the authors' exact preprocessing
(windowing / normalization / input size) is not published. We feed our standard
brain/subdural/bone 3-channel windows at the model's default input size with
ImageNet normalization — a common RSNA-2019 recipe, but scores may shift if their
pipeline differed. The 5 outputs are treated as ICH subtypes (order assumed);
study P(any) = max over them.
"""

from __future__ import annotations

import os

import numpy as np

# Assumed subtype order for the 5 outputs (documented assumption).
SUBTYPES_5 = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
BACKBONE = "convnextv2_base"
DEFAULT_CKPT = os.getenv("TRICORDER_ICH_CONVNEXT", "ConvNeXt_V2_Model.pth")


def _build_net():
    import timm
    import torch.nn as nn

    class ConvNeXtICH(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = timm.create_model(BACKBONE, pretrained=False, num_classes=0, in_chans=3)
            feat = self.model.num_features  # 1024
            self.head = nn.Sequential(
                nn.Linear(feat, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.0), nn.Linear(512, 5)
            )

        def forward(self, x):
            return self.head(self.model(x))

    return ConvNeXtICH()


class ConvNeXtICHModel:
    supports_gradcam = True
    untrained = False
    model_name = "RSNA-2019 ICH ConvNeXt V2-base (Kaggle, Apache-2.0) — real open weights"

    def __init__(self, ckpt_path: str = DEFAULT_CKPT):
        import torch

        from src import config

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        p = ckpt_path if os.path.isabs(ckpt_path) else str(config.REPO_ROOT / ckpt_path)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"ConvNeXt checkpoint not found at {p}. Download the Kaggle 'ConvNeXt V2 Model' "
                f"and place it there, or set TRICORDER_ICH_CONVNEXT."
            )
        self.net = _build_net().to(self.device).eval()
        self._load(p)
        # timm default input size for this backbone.
        cfg = self.net.model.default_cfg
        self.input_hw = int(cfg.get("input_size", (3, 224, 224))[-1])
        self.mean = np.array(cfg.get("mean", (0.485, 0.456, 0.406)), dtype=np.float32)
        self.std = np.array(cfg.get("std", (0.229, 0.224, 0.225)), dtype=np.float32)

    def _load(self, p):
        import torch

        sd = torch.load(p, map_location=self.device, weights_only=False)
        for k in ("state_dict", "model", "model_state_dict"):
            if isinstance(sd, dict) and k in sd:
                sd = sd[k]
                break
        sd = {kk.replace("module.", ""): vv for kk, vv in sd.items()}
        model_sd = self.net.state_dict()
        compatible = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
        frac = len(compatible) / max(1, len(model_sd))
        if frac < 0.9:
            raise RuntimeError(
                f"ConvNeXt checkpoint matched only {len(compatible)}/{len(model_sd)} params "
                f"({frac:.0%}) — architecture mismatch."
            )
        self.net.load_state_dict(compatible, strict=False)
        print(f"[ICH/convnext] loaded {len(compatible)}/{len(model_sd)} params ({frac:.0%}) from {os.path.basename(p)}")

    def _prep(self, three_ch_slice: np.ndarray):
        """(3,Y,X) [0,1] → normalized tensor (1,3,H,W)."""
        import torch
        import torch.nn.functional as F

        t = torch.from_numpy(three_ch_slice[None]).float().to(self.device)
        t = F.interpolate(t, size=(self.input_hw, self.input_hw), mode="bilinear", align_corners=False)
        mean = torch.tensor(self.mean, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=self.device).view(1, 3, 1, 1)
        return (t - mean) / std

    def score(self, three_ch: np.ndarray, batch: int = 8) -> tuple[np.ndarray, dict]:
        import torch

        z = three_ch.shape[0]
        probs = np.zeros((z, 5), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, z, batch):
                xs = torch.cat([self._prep(s) for s in three_ch[i : i + batch]], dim=0)
                probs[i : i + batch] = torch.sigmoid(self.net(xs)).cpu().numpy()
        per_slice_any = probs.max(axis=1)  # P(any) ≈ max subtype prob
        subtypes = {SUBTYPES_5[j]: round(float(probs[:, j].max()), 4) for j in range(5)}
        return per_slice_any, subtypes

    def gradcam(self, slice_3ch: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        # last depthwise conv in the ConvNeXt backbone
        last = None
        for m in self.net.model.modules():
            if isinstance(m, nn.Conv2d):
                last = m
        feats = {}
        h1 = last.register_forward_hook(lambda _m, _i, o: feats.__setitem__("a", o))
        h2 = last.register_full_backward_hook(lambda _m, _gi, go: feats.__setitem__("g", go[0]))
        try:
            x = self._prep(slice_3ch).requires_grad_(True)
            self.net.zero_grad(set_to_none=True)
            logits = self.net(x)
            logits[0, logits[0].argmax()].backward()
            a, g = feats["a"][0], feats["g"][0]
            cam = torch.relu((g.mean(dim=(1, 2), keepdim=True) * a).sum(0))
            cam = cam / (cam.max() + 1e-6)
            cam = F.interpolate(cam[None, None], size=slice_3ch.shape[1:], mode="bilinear", align_corners=False)
            return cam[0, 0].detach().cpu().numpy()
        finally:
            h1.remove()
            h2.remove()
