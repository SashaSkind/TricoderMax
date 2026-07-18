"""
Study-type routing accuracy on hand-labeled cases.

Loads labeled header/indication cases from eval/router_cases.csv (columns:
indication, StudyDescription, ContrastBolusAgent, expected) and reports accuracy
plus a confusion matrix over StudyType. Replace/extend the CSV with ~50 real
hand-labeled studies for the reported number.

    uv run python -m eval.router_eval
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from src.contract import StudyContext, StudyType
from src.router import classify

_DEFAULT = Path(__file__).resolve().parent / "router_cases.csv"


def _ctx(row: dict) -> StudyContext:
    return StudyContext(
        accession="EVAL", study_uid="1", n_slices=1, modality="CT",
        indication=row.get("indication") or None, study_type=StudyType.unknown,
        dicom_meta={
            "StudyDescription": row.get("StudyDescription", ""),
            "ContrastBolusAgent": row.get("ContrastBolusAgent", ""),
        },
    )


def run(cases_csv: str | Path = _DEFAULT) -> dict:
    rows = list(csv.DictReader(open(cases_csv)))
    correct = 0
    confusion: Counter = Counter()
    misses = []
    for r in rows:
        pred = classify(_ctx(r)).study_type.value
        exp = r["expected"].strip()
        confusion[(exp, pred)] += 1
        if pred == exp:
            correct += 1
        else:
            misses.append((r.get("StudyDescription", ""), exp, pred))
    n = len(rows)
    acc = correct / n if n else float("nan")
    return {"n": n, "accuracy": round(acc, 4), "confusion": confusion, "misses": misses}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(_DEFAULT))
    args = ap.parse_args()
    res = run(args.cases)
    print(f"Router accuracy: {res['accuracy']:.3f}  (n={res['n']})")
    print("\nConfusion (expected → predicted):")
    for (exp, pred), c in sorted(res["confusion"].items()):
        flag = "" if exp == pred else "  <-- miss"
        print(f"  {exp:>22} → {pred:<22} {c}{flag}")
    if res["misses"]:
        print("\nMisclassified:")
        for desc, exp, pred in res["misses"]:
            print(f"  {desc!r}: expected {exp}, got {pred}")


if __name__ == "__main__":
    main()
