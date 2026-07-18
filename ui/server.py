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


@app.post("/api/analyze")
def api_analyze(req: AnalyzeRequest) -> JSONResponse:
    """Run the REAL detector panel + Claude on a DICOM series dir. Slow on CPU."""
    path = req.path.strip()
    abs_path = (config.REPO_ROOT / path).resolve() if not os.path.isabs(path) else Path(path).resolve()
    if not abs_path.is_dir():
        raise HTTPException(404, f"not a directory: {path}")
    try:
        tri: TriageResult = triage_real_study(str(abs_path))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"{type(e).__name__}: {e}")

    def module_view(mid: str) -> dict | None:
        r = next((x for x in tri.panel.results if x.module_id == mid), None)
        if not r:
            return None
        d = r.model_dump()
        # overlay served from artifacts/<accession>/<name>
        d["overlay_urls"] = [
            f"/api/artifact/{tri.study.accession}/{str(p).split('/')[-1]}"
            for p in r.evidence.overlay_paths
        ]
        return d

    return JSONResponse(
        {
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
    )


@app.get("/api/artifact/{accession}/{name}")
def api_artifact(accession: str, name: str) -> FileResponse:
    # Confined to artifacts/<accession>/ — no path traversal.
    if "/" in name or ".." in name or ".." in accession:
        raise HTTPException(400, "bad path")
    path = (config.ARTIFACTS_DIR / accession / name).resolve()
    if not str(path).startswith(str(config.ARTIFACTS_DIR.resolve())) or not path.exists():
        raise HTTPException(404, "not found")
    return FileResponse(path)
