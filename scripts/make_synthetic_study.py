"""
Write a VALID synthetic DICOM head-CT series so you can test the real pipeline and
the UI's "Real study" tab without downloading 25 GB.

It is NOT a real patient — just correctly-formatted DICOM (HU, positions, spacing)
with a bright focal hyperdensity standing in for a bleed, so the real model + UI
run end-to-end. Real CQ500/RSNA studies slot into the same path.

    uv run python scripts/make_synthetic_study.py
    # → data/SYNTH-DEMO/series1/*.dcm ; point the UI "Real study" tab at it
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "data" / "SYNTH-DEMO" / "series1"


def _write(path, z_pos, instance, px_u16):
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = fm
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = pydicom.uid.CTImageStorage
    ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = "1.2.826.0.1.9999.SYNTH"
    ds.SeriesInstanceUID = "1.2.826.0.1.9999.SYNTH.1"
    ds.Modality = "CT"
    ds.PatientSex = "M"
    ds.PatientAge = "061Y"
    ds.AccessionNumber = "SYNTH-DEMO"
    ds.StudyDescription = "CT HEAD WO CONTRAST"
    ds.InstanceNumber = instance
    ds.ImagePositionPatient = [0.0, 0.0, float(z_pos)]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [0.5, 0.5]
    ds.SliceThickness = 5.0
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = -1024.0
    ds.Rows, ds.Columns = px_u16.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = px_u16.astype(np.uint16).tobytes()
    ds.save_as(str(path), write_like_original=False)


def main(n=24, size=256):
    OUT.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = size // 2
    skull = ((xx - cx) ** 2) / (110.0**2) + ((yy - cy) ** 2) / (120.0**2)
    for i in range(n):
        z = 100.0 - i * 5.0  # superior→inferior
        px = np.full((size, size), 1024, dtype=np.uint16)  # ~0 HU
        px[(skull > 0.9) & (skull < 1.05)] = 1024 + 900  # calvarium
        px[skull >= 1.05] = 24  # air outside (~ -1000 HU)
        # focal "bleed": bright hyperdensity (~+65 HU) on the mid slices, off-center
        if n // 3 <= i <= 2 * n // 3:
            px[cy - 18 : cy + 18, cx + 25 : cx + 61] = 1024 + 65
        _write(OUT / f"{i:04d}.dcm", z, i, px)
    print(f"wrote {n} slices → {OUT}")
    print(f"Point the UI 'Real study' tab (or run_study.py) at: data/SYNTH-DEMO/series1")


if __name__ == "__main__":
    main()
