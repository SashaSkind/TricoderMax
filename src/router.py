"""
Study-type routing — metadata only (invariant 2). Never inspects pixels.

Classifies a study from DICOM header fields (Modality, BodyPartExamined,
StudyDescription, ProtocolName, ContrastBolusAgent) plus the free-text clinical
indication. The failure mode is deliberately biased toward the SUPERSET: when the
type is unclear we return `unknown`, which makes the orchestrator run every panel
member (invariant 4 — waste compute, never skip a finding).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.contract import StudyContext, StudyType

_HEAD_TERMS = ("HEAD", "BRAIN", "SKULL", "CRANI", "NCCT", "CT H", "INTRACRAN", "NEURO")
_CONTRAST_TERMS = ("CONTRAST", "CTA", "ANGIO", "WITH IV", "W/ CONTRAST", "POST CONTRAST")
_NONCONTRAST_HINTS = ("WITHOUT CONTRAST", "W/O CONTRAST", "WO CONTRAST", "NON-CONTRAST", "NONCONTRAST", "NCCT")


@dataclass
class Routing:
    study_type: StudyType
    reason: str


def _text(ctx: StudyContext) -> str:
    parts = [ctx.indication or "", ctx.modality or ""]
    parts += list((ctx.dicom_meta or {}).values())
    return " ".join(parts).upper()


def classify(ctx: StudyContext) -> Routing:
    # Respect an explicitly-set type (e.g. a test or upstream override).
    if ctx.study_type != StudyType.unknown:
        return Routing(ctx.study_type, "study_type set upstream")

    text = _text(ctx)
    is_ct = "CT" in (ctx.modality or "").upper()
    has_head = any(term in text for term in _HEAD_TERMS)

    if not is_ct or not has_head:
        return Routing(
            StudyType.unknown,
            f"ambiguous (ct={is_ct}, head_terms={has_head}) → superset",
        )

    has_contrast = any(term in text for term in _CONTRAST_TERMS)
    has_noncontrast = any(term in text for term in _NONCONTRAST_HINTS)
    bolus = (ctx.dicom_meta or {}).get("ContrastBolusAgent", "").strip()

    if (has_contrast or bolus) and not has_noncontrast:
        return Routing(StudyType.head_ct_other, "head CT with contrast/CTA")
    return Routing(StudyType.head_ct_noncontrast, "non-contrast head CT")
