"""
Run the REAL triage pipeline on one DICOM study directory.

Unlike the Streamlit demo (which uses mock modules on synthetic volumes), this
forces the real detector panel + real Claude layer.

    uv run python run_study.py --study data/<StudyInstanceUID> [--accession ACC]

Requirements for a fully-real run:
  - real modules:  (default here) real DICOM in --study
  - ICH weights:   weights/ich/ich_cnn.pth  (else set TRICORDER_ICH_ALLOW_RANDOM=1)
  - Claude layer:  ANTHROPIC_API_KEY in .env  (else detector-only fallback)
"""

from __future__ import annotations

import argparse
import os

# Real detectors, not mocks — must be set before importing src modules.
os.environ.setdefault("TRICORDER_USE_MOCKS", "0")


def _license():
    print("=" * 72)
    print("NON-DIAGNOSTIC RESEARCH PROTOTYPE — prioritization only, never diagnosis.")
    print("Data is non-commercial use only. See LICENSE_NOTICE.md.")
    print("=" * 72)


def main():
    _license()
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, help="path to a single-series DICOM dir")
    ap.add_argument("--accession", default=None)
    args = ap.parse_args()

    from src import config
    from src.dicom_io import load_series, read_study_context
    from src.pipeline import triage_study

    print(f"Claude layer: {'ENABLED' if config.ANTHROPIC_API_KEY else 'fallback (no ANTHROPIC_API_KEY)'}")
    print(f"ICH weights:  {'present' if (config.WEIGHTS_DIR / 'ich' / 'ich_cnn.pth').exists() else 'MISSING (set TRICORDER_ICH_ALLOW_RANDOM=1 for a wiring test)'}")

    ctx = read_study_context(args.study, accession=args.accession)
    vol = load_series(args.study)
    result = triage_study(ctx, vol.hu)

    d = result.decision
    print("\n" + "=" * 72)
    print(f"STUDY {ctx.accession} · {ctx.study_type.value} · {vol.n_slices} slices")
    print(f"DECISION: {d.action.upper()} · band={d.priority_band} · score={d.priority_score}")
    print(f"reason: {d.reason}")
    print(f"routing: {result.panel.routing_reason}")
    print("\nDetector panel:")
    for r in result.panel.results:
        prov = (f"AUC={r.calibration.metric_value} on {r.calibration.eval_dataset}"
                if r.calibration else "uncalibrated")
        val = "—" if r.value is None else f"{r.value}{(' ' + r.units) if r.units else ''}"
        flags = " [approx]" if r.approximation else ""
        print(f"  {r.finding_class.value:<24} {r.status:<12} value={val:<10} "
              f"positive={r.positive} {prov}{flags}")
        if r.evidence.overlay_paths:
            print(f"      overlays: {r.evidence.overlay_paths}")
        if r.error:
            print(f"      ERROR: {r.error}")
    print(f"\nAssessment ({result.assessment.source}): {result.assessment.summary}")
    for c in result.assessment.caveats:
        print(f"  caveat: {c}")
    print("=" * 72)
    print("This study remains in the radiologist reading queue regardless of the above.")


if __name__ == "__main__":
    main()
