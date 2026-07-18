"""Phase 1: DICOM series load, HU conversion, ImagePositionPatient ordering."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pydicom")

from src.dicom_io import load_series, read_study_context


def test_slices_ordered_by_position_not_filename(synthetic_series):
    vol = load_series(synthetic_series["dir"])
    assert vol.n_slices == synthetic_series["n"]
    # z must be strictly descending (superior→inferior), regardless of filenames.
    assert vol.z_positions == sorted(vol.z_positions, reverse=True)
    assert vol.z_positions == synthetic_series["z_positions"]


def test_hu_conversion_applies_slope_intercept(synthetic_series):
    vol = load_series(synthetic_series["dir"])
    # Background stored 1024 with intercept -1024 → ~0 HU.
    assert abs(float(np.median(vol.hu))) < 5
    # Calvarial ring ~ +900 HU present.
    assert vol.hu.max() > 800
    # "Bleed" square is a few tens of HU above background, not saturated.
    center = vol.hu[:, 12:20, 12:20]
    assert 30 < float(center.mean()) < 60


def test_ordering_matches_encoded_slice_index(synthetic_series):
    """The central value encodes slice_idx; after sorting by z it must be monotonic."""
    vol = load_series(synthetic_series["dir"])
    centers = [float(vol.hu[k, 12:20, 12:20].mean()) for k in range(vol.n_slices)]
    # slice_idx 0 (z=100, top) has the lowest +val; increasing downward.
    assert centers == sorted(centers)


def test_read_study_context_metadata_only(synthetic_series):
    ctx = read_study_context(synthetic_series["dir"])
    assert ctx.modality == "CT"
    assert ctx.age == 58
    assert ctx.sex == "M"
    assert ctx.n_slices == synthetic_series["n"]
    assert "HEAD" in (ctx.indication or "").upper()
