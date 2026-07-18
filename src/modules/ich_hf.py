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
    supports_gradcam = False  # ViT attention attribution is out of scope here
    untrained = False

    def __init__(self, repo: str = DEFAULT_REPO):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(repo)
        self.model = AutoModelForImageClassification.from_pretrained(repo).to(self.device).eval()
        self.pos_idx = _positive_index(self.model.config.id2label)
        self.model_name = f"RSNA-2019 ViT ({repo}, HuggingFace) — real open weights, weak (~0.61 acc)"

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
