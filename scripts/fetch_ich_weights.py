"""
Fetch REAL RSNA-2019 ICH open weights from Kaggle into weights/ich/.

RSNA-2019 Intracranial Hemorrhage Detection was a public Kaggle competition, so
authors published their trained weights as Kaggle *datasets* attached to their
inference notebooks. This pulls one such dataset and lays it out for ICHModel.

Prereqs (one time):
  1. pip install kaggle   (already in the .[ml] extra? if not: uv pip install kaggle)
  2. Kaggle account -> Settings -> "Create New API Token" -> ~/.kaggle/kaggle.json
     (chmod 600), or set KAGGLE_USERNAME / KAGGLE_KEY.
  3. Accept the competition rules once:
     https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/rules

Usage:
  # Point at a public weights dataset (find the exact ref in a public RSNA-2019
  # inference notebook's "+ Add Data" panel — it lists the dataset it loads):
  python scripts/fetch_ich_weights.py --dataset <owner>/<slug> --backbone tf_efficientnet_b0

  # Then set the backbone to match and run real inference:
  TRICORDER_USE_MOCKS=0 TRICORDER_ICH_BACKBONE=<backbone> \
    python run_study.py --study data/<uid>

The loader (src/modules/ich_model.py) unwraps state_dict/model containers, strips
`module.` prefixes, and loads with strict=False, so a matching backbone loads
cleanly even if the head differs.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

DEST = Path(__file__).resolve().parent.parent / "weights" / "ich"


def _api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("kaggle not installed. `uv pip install kaggle` and set up ~/.kaggle/kaggle.json.")
    api = KaggleApi()
    api.authenticate()
    return api


def fetch(dataset: str, backbone: str):
    DEST.mkdir(parents=True, exist_ok=True)
    api = _api()
    print(f"↓ {dataset} -> {DEST}")
    api.dataset_download_files(dataset, path=str(DEST), unzip=True, quiet=False)

    ckpts = sorted(glob.glob(str(DEST / "**" / "*.pth"), recursive=True)
                   + glob.glob(str(DEST / "**" / "*.pt"), recursive=True))
    if not ckpts:
        sys.exit(f"No .pth/.pt found under {DEST} after download — check the dataset ref.")

    # ICHModel loads weights/ich/ich_cnn.pth by default; symlink/copy the first ckpt there.
    primary = DEST / "ich_cnn.pth"
    if not primary.exists():
        os.link(ckpts[0], primary) if hasattr(os, "link") else None
    manifest = {"dataset": dataset, "backbone": backbone, "checkpoints": [os.path.relpath(c, DEST) for c in ckpts],
                "primary": "ich_cnn.pth"}
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"✓ {len(ckpts)} checkpoint(s). Primary: {primary}")
    print(f"  Set TRICORDER_ICH_BACKBONE={backbone} to match this checkpoint.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Kaggle dataset ref: owner/slug")
    ap.add_argument("--backbone", default="tf_efficientnet_b0", help="timm backbone matching the checkpoint")
    args = ap.parse_args()
    fetch(args.dataset, args.backbone)
