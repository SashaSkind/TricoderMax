"""
Per-module evaluation: ROC / AUC + operating point, on CQ500 (external) or a
synthetic demo cohort.

Writes:
  registry_calibration.json  — calibration per module (the registry's single
                               source of truth, Flag F5). Loaded on next import.
  eval_results.json          — ROC points + operating point, consumed by the UI.
  artifacts/eval/<module>_roc.png (gitignored)

Real usage (needs CQ500 + ICH weights):
  uv run python -m eval.run_eval --manifest data/cq500_manifest.csv --dataset CQ500

Manifest CSV columns: accession, path (study dir), and one binary label column per
finding class evaluated, e.g. `intracranial_hemorrhage`, `midline_shift`.

Demo (no data): synthesizes correlated scores/labels so the whole harness runs and
populates calibration — labeled SYNTHETIC-DEMO, never claimed as CQ500.
  uv run python -m eval.run_eval --demo
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from eval.metrics import roc_curve, sens_spec_at
from src import config
from src.registry import REGISTRY

_CALIB_FILE = config.REPO_ROOT / "registry_calibration.json"
_RESULTS_FILE = config.REPO_ROOT / "eval_results.json"


def _license():
    print("=" * 72)
    print("NON-DIAGNOSTIC RESEARCH PROTOTYPE — data is non-commercial use only.")
    print("Reporting EXTERNAL metrics. See LICENSE_NOTICE.md.")
    print("=" * 72)


# ── score collection ─────────────────────────────────────────────────────────
def scores_from_manifest(manifest: str) -> dict[str, dict[str, np.ndarray]]:
    """Run each module over the manifest studies → {module_id: {scores, labels}}."""
    from src.dicom_io import load_series, read_study_context
    from src.modules.factory import build_module

    rows = list(csv.DictReader(open(manifest)))
    acc: dict[str, dict[str, list]] = {}
    for mid, desc in REGISTRY.items():
        fc = desc.finding_class.value
        if not any(fc in r for r in rows[:1]):
            continue
        scores, labels = [], []
        module = build_module(mid)
        for r in rows:
            if r.get(fc, "") == "":
                continue
            vol = load_series(r["path"])
            ctx = read_study_context(r["path"], accession=r.get("accession"))
            res = module.run(vol.hu, ctx)
            if res.status != "ok" or res.value is None:
                continue
            scores.append(float(res.value))
            labels.append(int(float(r[fc]) >= 0.5))
        if scores:
            acc[mid] = {"scores": np.array(scores), "labels": np.array(labels)}
    return acc


def scores_demo(seed: int = 0) -> dict[str, dict[str, np.ndarray]]:
    """Synthetic correlated cohort. Scores live on each module's real scale."""
    rng = np.random.default_rng(seed)
    n = 220
    out = {}
    # ICH: probability scale [0,1], target AUC ~0.9
    y = (rng.random(n) < 0.35).astype(int)
    pos = np.clip(rng.normal(0.72, 0.16, n), 0, 1)
    neg = np.clip(rng.normal(0.28, 0.16, n), 0, 1)
    out["ich_v1"] = {"scores": np.where(y == 1, pos, neg), "labels": y}
    # Midline: mm scale, threshold 5mm, target AUC ~0.85
    y2 = (rng.random(n) < 0.25).astype(int)
    pos2 = np.clip(rng.normal(8.5, 3.0, n), 0, None)
    neg2 = np.clip(rng.normal(2.2, 1.8, n), 0, None)
    out["midline_shift_v1"] = {"scores": np.where(y2 == 1, pos2, neg2), "labels": y2}
    return out


# ── evaluation ───────────────────────────────────────────────────────────────
def evaluate(collected: dict, dataset: str) -> tuple[dict, dict]:
    calibration, results = {}, {}
    for mid, data in collected.items():
        desc = REGISTRY[mid]
        thr = desc.threshold if desc.threshold is not None else float(np.median(data["scores"]))
        roc = roc_curve(data["scores"], data["labels"])
        sens, spec = sens_spec_at(data["scores"], data["labels"], thr)
        n = int(len(data["scores"]))
        calibration[mid] = {
            "eval_dataset": dataset, "metric": "auc", "metric_value": roc.auc, "n": n,
            "sensitivity_at_threshold": sens, "specificity_at_threshold": spec,
        }
        results[mid] = {
            "fpr": roc.fpr, "tpr": roc.tpr, "thresholds": roc.thresholds, "auc": roc.auc,
            "threshold": thr, "sensitivity": sens, "specificity": spec,
            "operating_fpr": round(1 - spec, 4), "operating_tpr": sens,
            "dataset": dataset, "n": n,
        }
        print(f"  {mid}: AUC={roc.auc}  sens={sens} spec={spec} @thr={thr} (n={n}, {dataset})")
    return calibration, results


def _save_roc_plots(results: dict):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    out = config.ARTIFACTS_DIR / "eval"
    out.mkdir(parents=True, exist_ok=True)
    for mid, r in results.items():
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(r["fpr"], r["tpr"], label=f"AUC={r['auc']}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.scatter([r["operating_fpr"]], [r["operating_tpr"]], color="red", zorder=5, label="operating point")
        ax.set_xlabel("FPR (1−specificity)")
        ax.set_ylabel("TPR (sensitivity)")
        ax.set_title(f"{mid} — {r['dataset']}")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"{mid}_roc.png", dpi=110)
        plt.close(fig)


def main():
    _license()
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="CSV: accession,path,<finding_class labels>")
    ap.add_argument("--demo", action="store_true", help="synthetic cohort (no data needed)")
    ap.add_argument("--dataset", default=None, help="dataset name for provenance")
    args = ap.parse_args()

    if args.demo or not args.manifest:
        dataset = args.dataset or "SYNTHETIC-DEMO"
        print(f"[demo] synthetic cohort ({dataset}) — NOT real CQ500 numbers")
        collected = scores_demo()
    else:
        dataset = args.dataset or "CQ500"
        collected = scores_from_manifest(args.manifest)

    calibration, results = evaluate(collected, dataset)
    _CALIB_FILE.write_text(json.dumps(calibration, indent=2))
    _RESULTS_FILE.write_text(json.dumps(results, indent=2))
    _save_roc_plots(results)
    print(f"wrote {_CALIB_FILE.name} and {_RESULTS_FILE.name}")


if __name__ == "__main__":
    main()
