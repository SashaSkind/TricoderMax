"""Phase 4: study-type routing from metadata + indication text."""

from __future__ import annotations

from src.contract import StudyContext, StudyType
from src.router import classify


def _ctx(**meta) -> StudyContext:
    indication = meta.pop("indication", None)
    return StudyContext(
        accession="R-1", study_uid="1", n_slices=20, modality=meta.pop("modality", "CT"),
        indication=indication, study_type=StudyType.unknown, dicom_meta=meta,
    )


def test_noncontrast_head_ct():
    r = classify(_ctx(StudyDescription="CT HEAD WO CONTRAST"))
    assert r.study_type == StudyType.head_ct_noncontrast


def test_contrast_head_ct_is_other():
    r = classify(_ctx(StudyDescription="CT ANGIOGRAM BRAIN", ContrastBolusAgent="Omnipaque"))
    assert r.study_type == StudyType.head_ct_other


def test_ambiguous_goes_unknown_superset():
    r = classify(_ctx(StudyDescription="CT ABDOMEN"))  # no head term
    assert r.study_type == StudyType.unknown

    r2 = classify(_ctx(modality="MR", StudyDescription="MRI BRAIN"))  # not CT
    assert r2.study_type == StudyType.unknown


def test_indication_text_provides_head_signal():
    r = classify(_ctx(indication="NCCT head, rule out intracranial hemorrhage"))
    assert r.study_type == StudyType.head_ct_noncontrast


def test_explicit_type_respected():
    ctx = StudyContext(
        accession="R-2", study_uid="1", n_slices=10, study_type=StudyType.head_ct_noncontrast
    )
    assert classify(ctx).study_type == StudyType.head_ct_noncontrast
