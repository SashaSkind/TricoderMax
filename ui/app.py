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

import json  # noqa: E402

import streamlit as st  # noqa: E402

from src import config  # noqa: E402
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


def _load_eval() -> dict:
    f = config.REPO_ROOT / "eval_results.json"
    return json.loads(f.read_text()) if f.exists() else {}


def calibration_tab() -> None:
    st.subheader("Per-module calibration & operating point")
    results = _load_eval()
    if not results:
        st.info("No eval results yet. Run: `uv run python -m eval.run_eval --demo`")
        return
    mid = st.selectbox("Module", list(results))
    r = results[mid]
    st.caption(
        f"Evaluated on **{r['dataset']}** (n={r['n']}). "
        + ("⚠️ synthetic demo cohort — not real CQ500 numbers." if "SYNTH" in r["dataset"] else "External metric.")
    )
    # Threshold slider → recompute operating point on the ROC.
    default_thr = float(r["threshold"])
    thr = st.slider(
        f"Operating threshold ({mid})", 0.0, max(1.0, default_thr * 2),
        value=default_thr, step=0.01,
    )
    # Recompute the operating point at the chosen threshold from stored ROC points.
    fpr, tpr, sens, spec, op = _operating_point(r, thr)
    c1, c2, c3 = st.columns(3)
    c1.metric("AUC", r["auc"])
    c2.metric("sensitivity", f"{sens:.2f}")
    c3.metric("specificity", f"{spec:.2f}")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(r["fpr"], r["tpr"], label=f"ROC (AUC={r['auc']})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.scatter([op[0]], [op[1]], color="red", zorder=5, label=f"operating pt @ {thr:.2f}")
    ax.set_xlabel("FPR (1−specificity)")
    ax.set_ylabel("TPR (sensitivity)")
    ax.set_title(f"{mid} — {r['dataset']}")
    ax.legend(loc="lower right", fontsize=8)
    st.pyplot(fig)
    st.caption(
        "Lowering the threshold raises sensitivity (catch more) at the cost of "
        "specificity (more pages). A study is never removed from the queue at any threshold."
    )


def _operating_point(r: dict, thr: float):
    """Exact ROC vertex at a decision threshold (positive iff score >= thr).

    ROC arrays are aligned with descending `thresholds` (starting at +inf). The
    operating point is the last vertex whose threshold is still >= thr.
    """
    import numpy as np

    fpr = np.array(r["fpr"])
    tpr = np.array(r["tpr"])
    thresholds = r.get("thresholds")
    if not thresholds:
        return fpr, tpr, r["sensitivity"], r["specificity"], (r["operating_fpr"], r["operating_tpr"])
    idx = 0
    for i, t in enumerate(thresholds):
        if t >= thr:
            idx = i
    sens = float(tpr[idx])
    spec = float(1.0 - fpr[idx])
    return fpr, tpr, sens, spec, (float(fpr[idx]), float(tpr[idx]))


def main() -> None:
    banner()
    st.title("🩺 Tricorder — head-CT triage")

    tab_work, tab_calib = st.tabs(["Worklist", "Calibration / operating point"])
    with tab_calib:
        calibration_tab()
    with tab_work:
        worklist_tab()


def worklist_tab() -> None:
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
