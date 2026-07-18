"""
Synthetic worklist for the Phase-0 demo.

These are metadata-only study contexts with a tiny dummy volume — NO real pixels,
NO patient data. They exist so the whole worklist → panel → policy → UI pipeline
runs on mock modules before any imaging data is present. Real studies (Phase 1+)
replace this by loading DICOM series from `data/`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from src.contract import StudyContext, StudyType

_BASE = datetime(2026, 7, 18, 9, 0, tzinfo=timezone.utc)


def _ctx(i: int, **kw) -> StudyContext:
    defaults = dict(
        accession=f"DEMO-{1000 + i}",
        study_uid=f"1.2.826.0.1.9999.{1000 + i}",
        study_type=StudyType.head_ct_noncontrast,
        n_slices=32,
        acquired_at=_BASE - timedelta(minutes=7 * i),
    )
    defaults.update(kw)
    return StudyContext(**defaults)


SAMPLE_STUDIES: list[StudyContext] = [
    _ctx(0, age=58, sex="M", indication="High-speed MVC, GCS 13, ? intracranial hemorrhage."),
    _ctx(1, age=34, sex="F", indication="Sudden thunderclap headache, ? SAH."),
    _ctx(2, age=71, sex="M", indication="Fall, on anticoagulation, confusion."),
    _ctx(3, age=45, sex="F", study_type=StudyType.unknown, indication="Head CT, protocol unclear."),
    _ctx(4, age=29, sex="M", indication="Assault, facial trauma, ? calvarial fracture."),
    _ctx(5, age=63, sex="F", indication="Known glioma, follow-up, ? mass effect."),
]


def dummy_volume(ctx: StudyContext) -> np.ndarray:
    """A tiny placeholder volume so modules have something to receive."""
    rng = np.random.default_rng(abs(hash(ctx.accession)) % (2**32))
    return rng.normal(30.0, 20.0, size=(ctx.n_slices, 64, 64)).astype(np.float32)
