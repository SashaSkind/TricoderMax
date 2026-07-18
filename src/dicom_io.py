"""
DICOM series → HU volume.

Loads a single-series directory, converts stored pixel values to Hounsfield
Units via RescaleSlope/RescaleIntercept, and orders slices by
ImagePositionPatient[2] — NEVER by filename (filenames are not reliably ordered,
and misordering silently corrupts every downstream module).

Also extracts a metadata-only `StudyContext` for routing (invariant 2: routing
never touches pixels).
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

from src.contract import StudyContext, StudyType


@dataclass
class CTVolume:
    hu: np.ndarray  # (Z, Y, X) float32 Hounsfield Units, ordered superior→inferior
    spacing: tuple[float, float, float]  # (z, y, x) mm
    z_positions: list[float]  # ImagePositionPatient[2] per slice, same order as hu
    study_uid: str
    series_uid: str

    @property
    def n_slices(self) -> int:
        return int(self.hu.shape[0])


def _series_files(series_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(series_dir, "*.dcm")))
    if not files:
        # Some exports omit the extension.
        files = [
            os.path.join(series_dir, f)
            for f in os.listdir(series_dir)
            if not f.startswith(".")
        ]
    if not files:
        raise FileNotFoundError(f"No DICOM files in {series_dir}")
    return files


def _slice_z(ds) -> float:
    """z from ImagePositionPatient[2]; fall back to SliceLocation / InstanceNumber."""
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) == 3:
        return float(ipp[2])
    if getattr(ds, "SliceLocation", None) is not None:
        return float(ds.SliceLocation)
    return float(getattr(ds, "InstanceNumber", 0))


def load_series(series_dir: str) -> CTVolume:
    import pydicom

    datasets = []
    for f in _series_files(series_dir):
        try:
            datasets.append(pydicom.dcmread(f))
        except Exception:  # noqa: BLE001 — skip non-image sidecar files
            continue
    if not datasets:
        raise FileNotFoundError(f"No readable DICOM images in {series_dir}")

    # Order superior→inferior: descending z. Head CT convention for display.
    datasets.sort(key=_slice_z, reverse=True)

    ref = datasets[0]
    slope = float(getattr(ref, "RescaleSlope", 1.0))
    intercept = float(getattr(ref, "RescaleIntercept", 0.0))
    py, px = (float(v) for v in getattr(ref, "PixelSpacing", [1.0, 1.0]))
    z_positions = [_slice_z(d) for d in datasets]
    if len(z_positions) > 1:
        pz = abs(z_positions[1] - z_positions[0]) or float(getattr(ref, "SliceThickness", 1.0))
    else:
        pz = float(getattr(ref, "SliceThickness", 1.0))

    hu = np.stack(
        [d.pixel_array.astype(np.float32) * slope + intercept for d in datasets], axis=0
    )
    return CTVolume(
        hu=hu,
        spacing=(pz, py, px),
        z_positions=z_positions,
        study_uid=str(getattr(ref, "StudyInstanceUID", "")),
        series_uid=str(getattr(ref, "SeriesInstanceUID", "")),
    )


def _age_to_int(value) -> int | None:
    """DICOM AgeString like '058Y' → 58."""
    if not value:
        return None
    s = str(value).strip()
    if s and s[0].isdigit():
        digits = "".join(c for c in s if c.isdigit())
        return int(digits) if digits else None
    return None


def read_study_context(series_dir: str, accession: str | None = None) -> StudyContext:
    """Metadata-only StudyContext from DICOM headers (no pixels)."""
    import pydicom

    ref = pydicom.dcmread(_series_files(series_dir)[0], stop_before_pixels=True)
    n = len(_series_files(series_dir))
    sex = str(getattr(ref, "PatientSex", "") or "").upper()[:1]
    sex = sex if sex in ("M", "F") else ("O" if sex else None)
    ps = getattr(ref, "PixelSpacing", [1.0, 1.0])
    ps_y, ps_x = (float(ps[0]), float(ps[1])) if len(ps) == 2 else (1.0, 1.0)
    # Filesystem-safe, UNIQUE accession (used in artifact paths/URLs). Generic
    # series folder names ("series", "thin") collide across studies, so fall back
    # to <study-folder>-<series-folder> (unique per study) then StudyInstanceUID.
    d = series_dir.rstrip("/")
    parent, base = os.path.basename(os.path.dirname(d)), os.path.basename(d)
    raw_acc = (
        str(getattr(ref, "AccessionNumber", "") or "").strip()
        or (f"{parent}-{base}" if parent else base)
        or str(getattr(ref, "StudyInstanceUID", "study"))
    )
    fallback_acc = "".join(c for c in raw_acc if c.isalnum() or c in "-_").lstrip("-_") or "study"
    return StudyContext(
        accession=accession or fallback_acc,
        study_uid=str(getattr(ref, "StudyInstanceUID", "")),
        study_type=StudyType.unknown,  # inferred by the router (Phase 4)
        modality=str(getattr(ref, "Modality", "CT")),
        n_slices=n,
        age=_age_to_int(getattr(ref, "PatientAge", None)),
        sex=sex,
        indication=str(getattr(ref, "StudyDescription", "") or "") or None,
        dicom_meta={
            "BodyPartExamined": str(getattr(ref, "BodyPartExamined", "") or ""),
            "StudyDescription": str(getattr(ref, "StudyDescription", "") or ""),
            "ProtocolName": str(getattr(ref, "ProtocolName", "") or ""),
            "ContrastBolusAgent": str(getattr(ref, "ContrastBolusAgent", "") or ""),
            "PixelSpacingY": str(ps_y),
            "PixelSpacingX": str(ps_x),
        },
    )
