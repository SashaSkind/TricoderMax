"""
Fetch ONE CQ500 study's best non-contrast series from the Kaggle mirror into a
friendly folder for the demo.

    python scripts/fetch_cq500_study.py --study CQ500CT101 --out CQ500-CT-101-negative

Pages the mirror's file list to find the study, picks the series with the most
slices preferring a plain/thin non-contrast acquisition, and downloads it to
data/<out>/series/*.dcm. Needs a Kaggle token (see fetch_ich_weights.md).
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
DATASET = "crawford/qureai-headct"


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _series_files(api, study: str, max_pages: int = 40) -> dict[str, list[str]]:
    key = f"{study} {study}/"
    token, series = None, defaultdict(list)
    seen = False
    for _ in range(max_pages):
        res = api.dataset_list_files(DATASET, page_token=token, page_size=200)
        for f in res.files:
            if key in f.name:
                seen = True
                series["/".join(f.name.split("/")[:-1])].append(f.name)
        token = res.next_page_token
        if not token:
            break
        # stop once we've passed this study's block
        if seen and not any(key in f.name for f in res.files):
            break
    return series


def _pick_series(series: dict[str, list[str]]) -> tuple[str, list[str]]:
    def score(item):
        name, files = item
        u = name.upper()
        contrast_penalty = -10_000 if any(w in u for w in ("CONTRAST", "ANGIO", "CTA", "POST")) else 0
        plain_bonus = 500 if any(w in u for w in ("PLAIN", "THIN", "NON")) else 0
        return contrast_penalty + plain_bonus + len(files)

    return max(series.items(), key=score)


def fetch(study: str, out: str):
    api = _api()
    series = _series_files(api, study)
    if not series:
        raise SystemExit(f"No files found for {study} (raise --max-pages if it's a later batch).")
    name, files = _pick_series(series)
    dest = DATA / out / "series"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"{study}: {len(series)} series; downloading '{name.split('/')[-1]}' ({len(files)} slices) → {dest}")
    for i, fn in enumerate(sorted(files)):
        api.dataset_download_file(DATASET, fn, path=str(dest), force=False, quiet=True)
        if i % 50 == 0:
            print(f"  {i}/{len(files)}")
    print(f"done → data/{out}/series ({len(files)} slices)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, help="mirror folder name, e.g. CQ500CT101")
    ap.add_argument("--out", required=True, help="friendly output dir under data/")
    args = ap.parse_args()
    fetch(args.study, args.out)
