"""
Central configuration: paths, operating thresholds, feature flags.

Thresholds live here (and in the registry's calibration) so there is one place to
adjust an operating point. The UI's threshold slider (Phase 6) overrides at runtime.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("TRICORDER_DATA", REPO_ROOT / "data"))
ARTIFACTS_DIR = Path(os.getenv("TRICORDER_ARTIFACTS", REPO_ROOT / "artifacts"))
WEIGHTS_DIR = Path(os.getenv("TRICORDER_WEIGHTS", REPO_ROOT / "weights"))

# Per-module timeout inside the orchestrator fan-out (Phase 4).
MODULE_TIMEOUT_S = float(os.getenv("TRICORDER_MODULE_TIMEOUT_S", "120"))

# Claude layer (Phase 5). Falls back to detector-only ranking when unset.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("TRICORDER_CLAUDE_MODEL", "claude-opus-4-8")

# Use MockModule everywhere instead of real detectors (Phase 0 demo insurance).
USE_MOCKS = os.getenv("TRICORDER_USE_MOCKS", "1").strip() not in ("", "0", "false", "False")


def ensure_dirs() -> None:
    for d in (DATA_DIR, ARTIFACTS_DIR, WEIGHTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
