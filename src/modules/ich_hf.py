"""
Hugging Face ICH backend — a REAL RSNA-2019-trained open-weight model.

Wraps `DifeiT/rsna-intracranial-hemorrhage-detection` (Apache-2.0), a ViT-base
fine-tuned on RSNA-2019. No Kaggle needed — `transformers` downloads it from the
Hub. It is a genuine learned model, but a weak one (~0.61 reported accuracy), so
results are marked accordingly and never dressed up as strong detection.

Interface matches the timm backend enough for `ich.py` to use either:
  .score(three_ch) -> (per_slice_any: np.ndarray[Z], subtypes: dict)
  .model_name, .untrained, .supports_gradcam
"""

from __future__ import annotations

import os

import numpy as np

DEFAULT_REPO = os.getenv("TRICORDER_ICH_HF_MODEL", "DifeiT/rsna-intracranial-hemorrhage-detection")


def _positive_index(id2label: dict) -> int:
    """Index of the 'hemorrhage-present' class."""
    for idx, label in id2label.items():
        l = str(label).lower()
        if any(k in l for k in ("hemor", "haemor", "positive", "yes", "abnormal", "ich")):
            return int(idx)
    # Binary head with unhelpful labels → assume class 1 is positive.
    return 1 if len(id2label) == 2 else int(max(id2label, key=lambda k: int(k)))


class HFICHModel:
    supports_gradcam = True  # via ViT attention rollout (see .gradcam)
    untrained = False

    def __init__(self, repo: str = DEFAULT_REPO):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(repo)
        # eager attention so output_attentions works (needed for the heatmap);
        # sdpa (the default) silently drops attentions.
        self.model = (
            AutoModelForImageClassification.from_pretrained(repo, attn_implementation="eager")
            .to(self.device)
            .eval()
        )
        self.pos_idx = _positive_index(self.model.config.id2label)
        self.model_name = f"RSNA-2019 ViT ({repo}, HuggingFace) — real open weights, weak (~0.61 acc)"

    def gradcam(self, slice_3ch: np.ndarray) -> np.ndarray:
        """ViT saliency for one slice → (Y,X) heatmap in [0,1].

        Uses the last-layer CLS→patch attention (averaged over heads), reshaped to
        the patch grid and upsampled — the standard ViT 'where did the model look'
        map. It shows what the model attended to, honestly reflecting a weak model.
        """
        import torch
        import torch.nn.functional as F

        img = (np.transpose(slice_3ch, (1, 2, 0)) * 255).astype(np.uint8)
        inputs = self.processor(images=[img], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_attentions=True)
        attn = out.attentions[-1][0]  # (heads, tokens, tokens)
        cls_to_patches = attn[:, 0, 1:].mean(0)  # (num_patches,)
        n = int(round(cls_to_patches.shape[0] ** 0.5))
        grid = cls_to_patches.reshape(1, 1, n, n)
        cam = F.interpolate(grid, size=slice_3ch.shape[1:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)
        return cam.cpu().numpy()

    def score(self, three_ch: np.ndarray, batch: int = 16) -> tuple[np.ndarray, dict]:
        """three_ch: (Z,3,Y,X) in [0,1] → per-slice P(hemorrhage). No subtypes."""
        import torch

        z = three_ch.shape[0]
        probs = np.zeros(z, dtype=np.float32)
        for i in range(0, z, batch):
            chunk = three_ch[i : i + batch]
            # processor wants HWC uint8 images; it handles resize + normalize.
            imgs = [(np.transpose(s, (1, 2, 0)) * 255).astype(np.uint8) for s in chunk]
            inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            p = torch.softmax(logits, dim=-1)[:, self.pos_idx].cpu().numpy()
            probs[i : i + batch] = p
        return probs, {}
