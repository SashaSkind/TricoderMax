"""Shared fixtures — synthetic DICOM series so Phase-1 tests need no real data."""

from __future__ import annotations

import numpy as np
import pytest


def _write_slice(path, *, z_pos, instance, pixels_u16, slope=1.0, intercept=-1024.0):
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
    ds.StudyInstanceUID = "1.2.3.study"
    ds.SeriesInstanceUID = "1.2.3.series"
    ds.Modality = "CT"
    ds.PatientSex = "M"
    ds.PatientAge = "058Y"
    ds.AccessionNumber = "ACC-1"
    ds.StudyDescription = "CT HEAD WO CONTRAST"
    ds.InstanceNumber = instance
    ds.ImagePositionPatient = [0.0, 0.0, float(z_pos)]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [0.5, 0.5]
    ds.SliceThickness = 5.0
    ds.RescaleSlope = slope
    ds.RescaleIntercept = intercept
    ds.Rows, ds.Columns = pixels_u16.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = pixels_u16.astype(np.uint16).tobytes()
    ds.save_as(str(path), write_like_original=False)


@pytest.fixture
def synthetic_series(tmp_path):
    """A 5-slice series written in SHUFFLED filename order.

    Each slice's stored value encodes its z so we can assert correct ordering
    after load. A central bright square (~+40 HU over background) stands in for a
    parenchymal bleed; a dense ring (~+900 HU) stands in for the calvarium.
    """
    pydicom = pytest.importorskip("pydicom")  # noqa: F841
    n, h, w = 5, 32, 32
    z_positions = [100.0, 80.0, 60.0, 40.0, 20.0]  # superior→inferior
    # Write files whose names do NOT match z order (filename must be ignored).
    filename_order = [2, 0, 4, 1, 3]
    for fname_idx, slice_idx in enumerate(filename_order):
        z = z_positions[slice_idx]
        # background ~0 HU stored as 1024 (intercept -1024)
        px = np.full((h, w), 1024, dtype=np.uint16)
        # calvarial ring ~ +900 HU
        px[0, :] = px[-1, :] = px[:, 0] = px[:, -1] = 1024 + 900
        # "bleed": central square brighter by z-dependent amount so we can check order
        val = 1024 + 40 + slice_idx  # +40..+44 HU
        px[12:20, 12:20] = val
        _write_slice(tmp_path / f"{fname_idx:03d}.dcm", z_pos=z, instance=slice_idx, pixels_u16=px)
    return {"dir": str(tmp_path), "z_positions": z_positions, "n": n}
