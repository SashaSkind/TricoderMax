# Tricorder — head-CT critical-finding triage prototype

> ⚠️ **RESEARCH PROTOTYPE — NON-DIAGNOSTIC.** This system **prioritizes** head-CT
> studies for radiologist attention. It does **not** diagnose and **never**
> removes a study from the reading queue — it can only move a study *earlier*.

Runs a panel of detectors on every non-contrast head CT, uses Claude *downstream*
of the detectors to reconcile their output with clinical context, and either
pages immediately or reorders the reading worklist. Every study is still read by
a radiologist.

## Load-bearing invariants

1. No study ever leaves the queue — the system may only move it earlier.
2. No LLM gating on pixels — Claude runs strictly downstream of the detector panel.
3. The panel runs unconditionally on every applicable study.
4. Ambiguous study type → run the superset of panels (waste compute, never skip).
5. Every numeric output carries provenance (module, eval dataset, threshold).
6. A prominent non-diagnostic banner appears in any UI.

## Data (non-commercial — see `LICENSE_NOTICE.md`)

- **RSNA 2019 ICH** — ~25k studies, slice-level labels for 5 ICH subtypes. Train/dev.
- **CQ500** — 491 studies, ICH + midline shift + mass effect + calvarial fracture. Eval.

Never commit imaging data or pixel artifacts. `data/` and `artifacts/` are gitignored.
Report the **external** (CQ500) metric everywhere; never the internal one alone.

## Quick start

```bash
uv sync                            # core deps
uv run uvicorn ui.server:app --port 8000   # web UI on mock modules → http://localhost:8000
uv run pytest                      # contract + orchestrator + policy + server tests
```

The UI is a **FastAPI backend + one self-contained HTML page** ([ui/server.py](ui/server.py),
[ui/static/index.html](ui/static/index.html)) — a thin viewer over the pipeline, not the product.

Run the **real** pipeline on a DICOM study (real detectors + Claude):

```bash
uv sync --extra hf --extra llm     # torch + transformers + anthropic
# put ANTHROPIC_API_KEY in .env; drop a series into data/<uid>/
uv run python run_study.py --study data/<StudyInstanceUID>
```

**ICH backend** (`TRICORDER_ICH_BACKEND`):
- `hf` (default) — real RSNA-2019 ViT from Hugging Face ([DifeiT](https://huggingface.co/DifeiT/rsna-intracranial-hemorrhage-detection), Apache-2.0), no Kaggle. Genuine open weights but weak (~0.61 acc) — reported honestly.
- `timm` — a local RSNA-2019 CNN checkpoint at `weights/ich/ich_cnn.pth` (stronger; fetch with `scripts/fetch_ich_weights.py`), with Grad-CAM overlays.

Eval (ROC/AUC + calibration): `uv run python -m eval.run_eval --demo`
(or `--manifest data/cq500_manifest.csv` for real CQ500).

## Architecture

```
Head CT (DICOM series)
  -> Preprocess & route   (HU conversion, 3 windows, study-type check — metadata only)
  -> Detector panel       (parallel, unconditional, uniform ModuleResult contract)
  -> Claude layer         (aggregate, verify against context, rank — downstream only)
  -> Policy split         (immediate page | worklist reorder — deterministic, audited)
  -> Radiologist          (reads every study, both branches)
```

See `src/contract.py` for the module contract that seams detectors to the Claude layer.
