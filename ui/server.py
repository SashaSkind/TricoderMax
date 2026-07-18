"""
Tricorder web UI — FastAPI backend + a single self-contained HTML page.

The dashboard is a thin viewer; all substance lives in src/ (contract,
orchestrator, policy, claude_layer, eval). This server just exposes the pipeline
over HTTP and serves ui/static/index.html.

    uv run uvicorn ui.server:app --reload --port 8000
    # open http://localhost:8000

Endpoints:
    GET /                       → the single-page UI
    GET /api/worklist           → ranked studies (runs the pipeline)
    GET /api/calibration        → eval_results.json (ROC + operating points)
    GET /api/artifact/{acc}/{f} → overlay PNGs under artifacts/ (evidence)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from src import config  # noqa: E402
from src.pipeline import TriageResult, rank_worklist, triage_real_study, triage_study  # noqa: E402
from src.sample_worklist import SAMPLE_STUDIES, dummy_volume  # noqa: E402

app = FastAPI(title="Tricorder triage")
STATIC = Path(__file__).resolve().parent / "static"

_LICENSE = (
    "Data: RSNA-2019 ICH + CQ500 — non-commercial research use only. "
    "External (CQ500) metrics reported. Research prototype, non-diagnostic."
)


def _row(rank: int, r: TriageResult) -> dict:
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
        "routing_reason": r.panel.routing_reason,
        "reason": r.decision.reason,
        "audit": r.decision.audit,
        "summary": r.assessment.summary,
        "caveats": r.assessment.caveats,
        "assessment_source": r.assessment.source,
        "results": [m.model_dump() for m in r.panel.results],
    }


def _worklist() -> list[dict]:
    triaged = [triage_study(c, dummy_volume(c)) for c in SAMPLE_STUDIES]
    ranked = rank_worklist(triaged)
    return [_row(i, r) for i, r in enumerate(ranked, start=1)]


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text()


@app.get("/api/worklist")
def api_worklist() -> JSONResponse:
    return JSONResponse({"license": _LICENSE, "studies": _worklist()})


@app.get("/api/calibration")
def api_calibration() -> JSONResponse:
    f = config.REPO_ROOT / "eval_results.json"
    data = json.loads(f.read_text()) if f.exists() else {}
    return JSONResponse(data)


class AnalyzeRequest(BaseModel):
    path: str  # server-side path to a single-series DICOM directory
    force: bool = False  # re-run even if a cached result exists


# Friendly labels for known demo studies (shown in the clickable list).
_STUDY_LABELS = {
    "CQ500-CT-10-bleed": "CQ500-CT-10 · real bleed (radiologist ICH 0.98)",
    "SYNTH-DEMO": "Synthetic demo study (no real patient)",
}


def _series_dirs() -> list[Path]:
    """Every directory under data/ that holds DICOM slices (skips _-prefixed dirs)."""
    out = []
    if not config.DATA_DIR.exists():
        return out
    for d in sorted(config.DATA_DIR.rglob("*")):
        if not d.is_dir() or any(part.startswith("_") for part in d.relative_to(config.DATA_DIR).parts):
            continue
        if any(d.glob("*.dcm")):
            out.append(d)
    return out


def _cache_path(accession: str) -> Path:
    return config.ARTIFACTS_DIR / accession / "analysis.json"


@app.get("/api/studies")
def api_studies() -> JSONResponse:
    """Discover available DICOM studies so the UI can list them as clickable cards."""
    from src.dicom_io import read_study_context

    studies = []
    for d in _series_dirs():
        try:
            ctx = read_study_context(str(d))
        except Exception:  # noqa: BLE001
            continue
        top = d.relative_to(config.DATA_DIR).parts[0]
        studies.append(
            {
                "path": str(d.relative_to(config.REPO_ROOT)),
                "label": _STUDY_LABELS.get(top, top),
                "accession": ctx.accession,
                "n_slices": ctx.n_slices,
                "cached": _cache_path(ctx.accession).exists(),
            }
        )
    return JSONResponse({"studies": studies})


def _analyze_payload(tri: TriageResult) -> dict:
    def module_view(mid: str) -> dict | None:
        r = next((x for x in tri.panel.results if x.module_id == mid), None)
        if not r:
            return None
        d = r.model_dump()
        d["overlay_urls"] = [
            f"/api/artifact/{tri.study.accession}/{str(p).split('/')[-1]}"
            for p in r.evidence.overlay_paths
        ]
        return d

    return {
        "accession": tri.study.accession,
        "study_type": tri.study.study_type.value,
        "n_slices": tri.study.n_slices,
        "decision": {
            "action": tri.decision.action,
            "band": tri.decision.priority_band,
            "score": tri.decision.priority_score,
            "reason": tri.decision.reason,
        },
        "assessment": {"source": tri.assessment.source, "summary": tri.assessment.summary,
                       "caveats": tri.assessment.caveats},
        "ich": module_view("ich_v1"),
        "midline": module_view("midline_shift_v1"),
    }


@app.post("/api/analyze")
def api_analyze(req: AnalyzeRequest) -> JSONResponse:
    """Run the REAL detector panel + Claude on a DICOM series dir.

    Results are cached to artifacts/<accession>/analysis.json so a demo click is
    instant after the first (slow, ~2 min on CPU) run. Pass force=true to re-run.
    """
    path = req.path.strip()
    abs_path = (config.REPO_ROOT / path).resolve() if not os.path.isabs(path) else Path(path).resolve()
    if not abs_path.is_dir():
        raise HTTPException(404, f"not a directory: {path}")

    # Fast path: return the cached analysis if present.
    from src.dicom_io import read_study_context

    try:
        acc = read_study_context(str(abs_path)).accession
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"cannot read study: {e}")
    cache = _cache_path(acc)
    if cache.exists() and not req.force:
        payload = json.loads(cache.read_text())
        payload["cached"] = True
        return JSONResponse(payload)

    try:
        tri: TriageResult = triage_real_study(str(abs_path))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    payload = _analyze_payload(tri)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload))
    payload["cached"] = False
    return JSONResponse(payload)


@app.get("/api/artifact/{accession}/{name}")
def api_artifact(accession: str, name: str) -> FileResponse:
    # Confined to artifacts/<accession>/ — no path traversal.
    if "/" in name or ".." in name or ".." in accession:
        raise HTTPException(400, "bad path")
    path = (config.ARTIFACTS_DIR / accession / name).resolve()
    if not str(path).startswith(str(config.ARTIFACTS_DIR.resolve())) or not path.exists():
        raise HTTPException(404, "not found")
    return FileResponse(path)
