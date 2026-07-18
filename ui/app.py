"""
Streamlit worklist + study detail.

Phase 0: runs entirely on mock modules — a ranked worklist with per-module findings,
provenance, approximation/uncalibrated/failure indicators, and the mandatory
non-diagnostic banner. No ML, no real pixels.

Run:  uv run streamlit run ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from src.contract import ModuleResult  # noqa: E402
from src.pipeline import TriageResult, rank_worklist, triage_study  # noqa: E402
from src.sample_worklist import SAMPLE_STUDIES, dummy_volume  # noqa: E402

st.set_page_config(page_title="Tricorder triage", layout="wide")

_BAND_COLOR = {"immediate": "#c0392b", "urgent": "#d68910", "routine": "#5d6d7e"}


def banner() -> None:
    st.markdown(
        """
        <div style="background:#7b241c;color:#fff;padding:10px 16px;border-radius:6px;
                    font-weight:700;letter-spacing:.3px;margin-bottom:8px;">
          ⚠️ RESEARCH PROTOTYPE — NON-DIAGNOSTIC · PRIORITIZATION ONLY ·
          NEVER REMOVES A STUDY FROM THE READING QUEUE
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Data: RSNA-2019 ICH + CQ500 — non-commercial research use only. "
        "External (CQ500) metrics are reported. See LICENSE_NOTICE.md."
    )


@st.cache_data(show_spinner="Running detector panel…")
def _run_all() -> list[dict]:
    results = [triage_study(ctx, dummy_volume(ctx)) for ctx in SAMPLE_STUDIES]
    ranked = rank_worklist(results)
    # Cache-friendly: return plain dicts of what the UI needs.
    return [_to_row(i, r) for i, r in enumerate(ranked, start=1)]


def _to_row(rank: int, r: TriageResult) -> dict:
    return {
        "rank": rank,
        "accession": r.study.accession,
        "action": r.decision.action,
        "band": r.decision.priority_band,
        "score": r.decision.priority_score,
        "study_type": r.study.study_type.value,
        "indication": r.study.indication or "",
        "age": r.study.age,
        "sex": r.study.sex,
        "n_failed": len(r.panel.modules_failed),
        "superset": r.panel.superset_run,
        "reason": r.decision.reason,
        "routing_reason": r.panel.routing_reason,
        "audit": r.decision.audit,
        "summary": r.assessment.summary,
        "caveats": r.assessment.caveats,
        "assessment_source": r.assessment.source,
        "results": [m.model_dump() for m in r.panel.results],
    }


def _finding_line(m: dict) -> None:
    mr = ModuleResult(**m)
    tag = []
    if mr.status != "ok":
        tag.append(f"🚫 {mr.status}")
    elif mr.positive:
        tag.append("🔴 POSITIVE")
    else:
        tag.append("🟢 negative")
    if mr.approximation:
        tag.append("≈ approximation")
    if mr.calibration is None and mr.status == "ok":
        tag.append("⚠ uncalibrated")

    val = "—" if mr.value is None else f"{mr.value}{(' ' + mr.units) if mr.units else ''}"
    st.markdown(f"**{mr.finding_class.value}** · {mr.module_id} v{mr.module_version} — {' · '.join(tag)}")
    cols = st.columns([1, 1, 1, 2])
    cols[0].metric("value", val)
    cols[1].metric("threshold", "—" if mr.threshold is None else str(mr.threshold))
    cols[2].metric("runtime", f"{mr.runtime_ms} ms")
    # Provenance (invariant 5)
    if mr.calibration:
        c = mr.calibration
        cols[3].write(
            f"provenance: {c.metric.upper()}={c.metric_value} on {c.eval_dataset} (n={c.n})"
        )
    else:
        cols[3].write("provenance: **uncalibrated** — threshold not yet evaluated")
    if mr.status == "failed":
        st.error(f"module error: {mr.error}")
    if mr.evidence.slice_indices:
        st.caption(f"evidence slices: {mr.evidence.slice_indices}  ·  {mr.evidence.note or ''}")


def main() -> None:
    banner()
    st.title("🩺 Tricorder — head-CT triage worklist")

    rows = _run_all()
    paged = sum(1 for r in rows if r["action"] == "page")
    st.write(f"**{len(rows)} studies** · {paged} paged · every study remains in the reading queue.")

    left, right = st.columns([2, 3])
    with left:
        st.subheader("Worklist (ranked)")
        for r in rows:
            color = _BAND_COLOR.get(r["band"], "#5d6d7e")
            badge = "PAGE" if r["action"] == "page" else "reorder"
            flags = ""
            if r["n_failed"]:
                flags += f" · ⚠{r['n_failed']} failed"
            if r["superset"]:
                flags += " · superset"
            label = (
                f"#{r['rank']} · {r['accession']} · "
                f"{badge} · {r['band']} ({r['score']:.2f}){flags}"
            )
            if st.button(label, key=f"sel-{r['accession']}", use_container_width=True):
                st.session_state["sel"] = r["accession"]
            st.markdown(
                f"<div style='height:4px;background:{color};margin:-6px 0 10px;border-radius:2px;'></div>",
                unsafe_allow_html=True,
            )

    with right:
        sel = st.session_state.get("sel", rows[0]["accession"] if rows else None)
        row = next((r for r in rows if r["accession"] == sel), None)
        if not row:
            st.info("Select a study.")
            return
        st.subheader(f"Study {row['accession']}")
        st.write(
            f"{row['study_type']} · age {row['age']} · {row['sex']} · "
            f"indication: *{row['indication']}*"
        )
        band_c = _BAND_COLOR.get(row["band"], "#5d6d7e")
        st.markdown(
            f"<span style='background:{band_c};color:#fff;padding:3px 10px;border-radius:4px;'>"
            f"{row['action'].upper()} · {row['band']} · score {row['score']:.2f}</span>",
            unsafe_allow_html=True,
        )
        st.write(f"**Summary:** {row['summary']}")
        st.caption(f"ranking source: {row['assessment_source']}")
        if row["caveats"]:
            for c in row["caveats"]:
                st.warning(c)

        st.markdown("### Detector panel")
        for m in row["results"]:
            with st.container(border=True):
                _finding_line(m)

        with st.expander("Policy audit trail"):
            st.write(f"**Routing:** {row['routing_reason']}")
            st.write(f"**Decision:** {row['reason']}")
            for line in row["audit"]:
                st.text(f"• {line}")


if __name__ == "__main__":
    main()
else:
    main()
