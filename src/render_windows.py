"""
Render the three clinical windows for a study to PNGs (Phase-1 acceptance aid).

    uv run python -m src.render_windows --study data/<StudyUID> --slice 15

Writes brain/subdural/bone PNGs under artifacts/<study>/ (gitignored — no pixel
artifacts are committed). Prints the data license notice on startup.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src import config
from src.dicom_io import load_series
from src.windowing import WINDOWS, to_uint8, window_named


def _print_license() -> None:
    notice = config.REPO_ROOT / "LICENSE_NOTICE.md"
    print("=" * 72)
    print("NON-DIAGNOSTIC RESEARCH PROTOTYPE — data is non-commercial use only.")
    if notice.exists():
        print(f"See {notice}")
    print("=" * 72)


def main() -> None:
    _print_license()
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, help="path to a single-series DICOM dir")
    ap.add_argument("--slice", type=int, default=None, help="slice index (default: middle)")
    args = ap.parse_args()

    from PIL import Image

    vol = load_series(args.study)
    z = args.slice if args.slice is not None else vol.n_slices // 2
    z = max(0, min(z, vol.n_slices - 1))
    sl = vol.hu[z]

    out_dir = config.ARTIFACTS_DIR / Path(args.study).name
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in WINDOWS:
        img = to_uint8(window_named(sl, name))
        path = out_dir / f"{name}_slice{z}.png"
        Image.fromarray(img).save(path)
        print(f"wrote {path}  (window {name} = WL{WINDOWS[name][0]}/WW{WINDOWS[name][1]})")
    print(f"volume: {vol.n_slices} slices, spacing(z,y,x)={vol.spacing}, HU range "
          f"[{vol.hu.min():.0f}, {vol.hu.max():.0f}]")


if __name__ == "__main__":
    main()
