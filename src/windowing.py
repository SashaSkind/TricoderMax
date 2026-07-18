"""
CT windowing → normalized display / model channels.

Three clinical windows for non-contrast head CT:
  - brain    (WL 40,  WW 80)   — parenchyma, most ICH is conspicuous here
  - subdural  (WL 75,  WW 215) — thin extra-axial blood along the falx/tentorium
  - bone     (WL 600, WW 2800) — calvarium / fracture

The 3-channel stack (brain, subdural, bone) is the standard input the reference
ICH checkpoints expect, so this is also what Phase 2 feeds the model.
"""

from __future__ import annotations

import numpy as np

# (window level, window width) in HU.
WINDOWS: dict[str, tuple[float, float]] = {
    "brain": (40.0, 80.0),
    "subdural": (75.0, 215.0),
    "bone": (600.0, 2800.0),
}


def apply_window(hu: np.ndarray, wl: float, ww: float) -> np.ndarray:
    """Window HU to [0, 1] float32."""
    lo = wl - ww / 2.0
    hi = wl + ww / 2.0
    return np.clip((hu - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def window_named(hu: np.ndarray, name: str) -> np.ndarray:
    if name not in WINDOWS:
        raise KeyError(f"unknown window {name!r}; known: {list(WINDOWS)}")
    return apply_window(hu, *WINDOWS[name])


def three_channel(hu: np.ndarray) -> np.ndarray:
    """Stack brain/subdural/bone as channels.

    Input (Z, Y, X) → output (Z, 3, Y, X) with channel order (brain, subdural, bone).
    A single (Y, X) slice → (3, Y, X).
    """
    chans = [window_named(hu, n) for n in ("brain", "subdural", "bone")]
    axis = 1 if hu.ndim == 3 else 0
    return np.stack(chans, axis=axis)


def to_uint8(norm01: np.ndarray) -> np.ndarray:
    """A [0,1] windowed array → uint8 for PNG rendering."""
    return (np.clip(norm01, 0.0, 1.0) * 255.0).round().astype(np.uint8)
