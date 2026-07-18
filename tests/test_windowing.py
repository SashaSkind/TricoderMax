"""Phase 1: clinical windows and the 3-channel stack."""

from __future__ import annotations

import numpy as np

from src.windowing import WINDOWS, apply_window, three_channel, to_uint8, window_named


def test_window_maps_to_unit_range():
    hu = np.array([[-1000, 0, 40, 80, 3000]], dtype=np.float32)
    out = window_named(hu, "brain")  # WL40/WW80 → [0,80] HU maps to [0,1]
    assert out.min() == 0.0 and out.max() == 1.0
    # 40 HU is window center → 0.5
    assert abs(float(out[0, 2]) - 0.5) < 1e-6


def test_brain_vs_bone_contrast_differs():
    # A parenchymal-ish value is mid-gray on brain but much darker on bone.
    hu = np.full((8, 8), 45.0, dtype=np.float32)
    brain = window_named(hu, "brain").mean()
    bone = window_named(hu, "bone").mean()
    assert brain > 0.5 and brain > bone + 0.2

    # A dense calvarial value saturates bone but clips high on brain.
    dense = np.full((8, 8), 900.0, dtype=np.float32)
    assert window_named(dense, "bone").mean() > 0.4
    assert window_named(dense, "brain").mean() == 1.0


def test_three_channel_shapes_and_order():
    vol = np.zeros((4, 16, 16), dtype=np.float32)
    stk = three_channel(vol)
    assert stk.shape == (4, 3, 16, 16)  # (Z, C, Y, X)
    sl = three_channel(vol[0])
    assert sl.shape == (3, 16, 16)


def test_bleed_visible_on_brain_window(synthetic_series):
    """Phase-1 acceptance (synthetic proxy): the 'bleed' is conspicuous on brain."""
    from src.dicom_io import load_series

    vol = load_series(synthetic_series["dir"])
    sl = vol.hu[vol.n_slices // 2]
    brain = window_named(sl, "brain")
    bone = window_named(sl, "bone")
    center = brain[12:20, 12:20].mean()
    background = brain[4:8, 4:8].mean()
    assert center > background  # bleed brighter than parenchyma on brain window
    # Calvarial ring resolves on bone window.
    assert bone[0, :].mean() > 0.4


def test_to_uint8_range():
    x = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    assert list(to_uint8(x)) == [0, 128, 255]
