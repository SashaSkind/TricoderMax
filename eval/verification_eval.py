"""
Verification-value eval — does the Claude layer actually help?

The question the spec calls "the number that justifies the layer": on a labeled
set, does applying Claude's per-finding confidence_adjustment improve PRECISION at
a FIXED RECALL compared to the detector threshold alone? If Claude's down-weights
remove false positives without dropping true positives, precision@recall rises.

    # real (needs a labeled manifest with DICOM paths + ANTHROPIC_API_KEY):
    uv run python -m eval.verification_eval --manifest data/cq500_manifest.csv --recall 0.9

    # demo (no data/API): shows the metric on a synthetic cohort where Claude
    # correctly down-weights artifactual false positives.
    uv run python -m eval.verification_eval --demo

Scores the Claude LAYER, separately from the detectors. Reports EXTERNAL numbers.
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from eval.metrics import precision_at_recall


def _adjusted(det: np.ndarray, adj: np.ndarray) -> np.ndarray:
    """Detector prob + Claude confidence adjustment, clipped to [0,1]."""
    return np.clip(det + adj, 0.0, 1.0)


def report(det: np.ndarray, adj: np.ndarray, labels: np.ndarray, recall: float, dataset: str) -> dict:
    p_det, thr_det = precision_at_recall(det, labels, recall)
    p_adj, thr_adj = precision_at_recall(_adjusted(det, adj), labels, recall)
    delta = round(p_adj - p_det, 4)
    print(f"Verification value on {dataset} (n={len(labels)}, recall≥{recall}):")
    print(f"  detector alone       : precision={p_det} @thr={thr_det}")
    print(f"  detector + Claude adj : precision={p_adj} @thr={thr_adj}")
    print(f"  Δprecision           : {delta:+}  ← the number that justifies the layer")
    return {"precision_detector": p_det, "precision_with_claude": p_adj, "delta": delta,
            "recall": recall, "n": int(len(labels)), "dataset": dataset}


# ── real path ────────────────────────────────────────────────────────────────
def from_manifest(manifest: str, recall: float) -> dict:
    from src.claude_layer import assess
    from src.pipeline import triage_real_study

    rows = [r for r in csv.DictReader(open(manifest)) if r.get("path") and r.get("intracranial_hemorrhage") != ""]
    det, adj, labels = [], [], []
    for r in rows:
        tri = triage_real_study(r["path"])  # runs detectors + Claude
        ich = next((x for x in tri.panel.results if x.module_id == "ich_v1"), None)
        if not ich or ich.value is None:
            continue
        a = tri.assessment
        adjustment = next((v.confidence_adjustment for v in a.verification if v.module_id == "ich_v1"), 0.0)
        det.append(float(ich.value))
        adj.append(float(adjustment))
        labels.append(int(float(r["intracranial_hemorrhage"]) >= 0.5))
    if not det:
        sys.exit("No usable rows (need DICOM paths + ICH labels).")
    return report(np.array(det), np.array(adj), np.array(labels), recall, "CQ500")


# ── demo path ────────────────────────────────────────────────────────────────
def demo(recall: float, seed: int = 0) -> dict:
    """Synthetic cohort: detector has some artifactual FPs (high score, negative
    label); Claude down-weights exactly those. Shows precision@recall improving."""
    rng = np.random.default_rng(seed)
    n = 300
    labels = (rng.random(n) < 0.35).astype(int)
    det = np.where(labels == 1, rng.normal(0.8, 0.12, n), rng.normal(0.3, 0.18, n)).clip(0, 1)
    # 20% of negatives are artifactual false positives with a high detector score.
    fp = (labels == 0) & (rng.random(n) < 0.2)
    det[fp] = rng.uniform(0.7, 0.95, fp.sum())
    # Claude correctly down-weights the artifacts (and only those).
    adj = np.zeros(n)
    adj[fp] = -0.6
    print("[demo] synthetic cohort — Claude down-weights artifactual false positives")
    return report(det, adj, labels, recall, "SYNTHETIC-DEMO")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--recall", type=float, default=0.9)
    args = ap.parse_args()
    if args.demo or not args.manifest:
        demo(args.recall)
    else:
        from_manifest(args.manifest, args.recall)


if __name__ == "__main__":
    main()
