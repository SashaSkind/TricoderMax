"""
Prepare CQ500 studies for Tricorder.

CQ500 (qure.ai, CC BY-NC-SA — non-commercial). The original qure.ai download page
(headctstudy.qure.ai) is DEFUNCT (domain no longer resolves), so use a mirror:

  - Kaggle (recommended — you likely have a token):
        kaggle datasets download -d crawford/qureai-headct -p data/_cq500 --unzip
        python scripts/fetch_cq500.py --src data/_cq500
  - Academic Torrents:
        https://academictorrents.com/details/47e9d8aab761e75fd0a81982fa62bddf3a173831
        (download, then --src the extracted folder)
  - If you still have a direct zip URL from somewhere: --url "<zip_url>"

Whatever the source, this lays each DICOM series out as
  data/<study>/<series>/*.dcm  so run_study.py can consume it, and converts the
CQ500 reads CSV into an eval manifest with majority-vote labels.

Usage:
  python scripts/fetch_cq500.py --src data/_cq500          # reorganize a local dir
  python scripts/fetch_cq500.py --url "<zip_url>"          # download+reorganize a zip
  python scripts/fetch_cq500.py --reads path/to/reads.csv  # build eval manifest

Non-commercial research use only. No imaging data is committed (data/ gitignored).
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import shutil
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


# ── imaging ──────────────────────────────────────────────────────────────────
def _download(url: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    name = url.split("/")[-1].split("?")[0] or "cq500.zip"
    out = dest / name
    print(f"↓ {url}")
    with urllib.request.urlopen(url) as r, open(out, "wb") as f:  # noqa: S310 (user-supplied URL)
        shutil.copyfileobj(r, f)
    print(f"  saved {out} ({out.stat().st_size // (1024*1024)} MB)")
    return out


def _reorganize_dir(src: Path) -> int:
    """Lay every DICOM under `src` out as data/<study>/<series>/*.dcm by header UID."""
    import pydicom

    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for p in src.rglob("*"):
        if p.is_file():
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True)
                groups[(str(ds.StudyInstanceUID), str(ds.SeriesInstanceUID))].append(p)
            except Exception:  # noqa: BLE001 — skip non-DICOM
                continue

    for (study, series), files in groups.items():
        # Short, filesystem-safe names; run_study.py points at the series dir.
        sdir = DATA / f"CQ500-{study[-8:]}" / series[-8:]
        sdir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(sorted(files)):
            shutil.copy(f, sdir / f"{i:04d}.dcm")
        print(f"  {sdir.relative_to(DATA)}: {len(files)} slices")
    return len(groups)


def _reorganize(zip_path: Path) -> None:
    tmp = DATA / "_cq500_raw"
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    _reorganize_dir(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


# ── labels → eval manifest ───────────────────────────────────────────────────
# CQ500 reads.csv columns are prefixed by reader (R1/R2/R3). We majority-vote the
# fields Tricorder's modules produce labels for.
_LABEL_MAP = {
    "intracranial_hemorrhage": ["ICH"],
    "midline_shift": ["MidlineShift"],
    "mass_effect": ["MassEffect"],
    "calvarial_fracture": ["CalvarialFracture", "Fracture"],
}


def _majority(row: dict, bases: list[str]) -> int | None:
    votes = []
    for reader in ("R1", "R2", "R3"):
        for base in bases:
            key = f"{reader}:{base}"
            if key in row and row[key] != "":
                votes.append(int(float(row[key])))
    if not votes:
        return None
    return 1 if sum(votes) * 2 >= len(votes) else 0


def build_manifest(reads_csv: str) -> None:
    rows = list(csv.DictReader(open(reads_csv)))
    out = DATA / "cq500_manifest.csv"
    fields = ["accession", "path"] + list(_LABEL_MAP)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            name = r.get("name") or r.get("Name") or ""
            rec = {"accession": name, "path": ""}  # fill path after imaging is laid out
            for fc, bases in _LABEL_MAP.items():
                v = _majority(r, bases)
                rec[fc] = "" if v is None else v
            w.writerow(rec)
    print(f"wrote {out} ({len(rows)} studies). Fill the 'path' column with each "
          f"study's series dir under data/ before running eval.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="reorganize an already-downloaded CQ500 dir (Kaggle/torrent)")
    ap.add_argument("--url", action="append", default=[], help="CQ500 study zip URL (repeatable)")
    ap.add_argument("--reads", help="path to CQ500 reads.csv → build eval manifest")
    args = ap.parse_args()
    if not args.url and not args.reads and not args.src:
        ap.print_help()
        sys.exit(0)
    if args.src:
        n = _reorganize_dir(Path(args.src))
        print(f"reorganized {n} series into {DATA}/")
    for u in args.url:
        z = _download(u, DATA / "_cq500_zips")
        _reorganize(z)
    if args.reads:
        build_manifest(args.reads)
