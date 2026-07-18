# What we built vs. what we reused

Hackathon compliance: this file draws a hard line between original work created
during the event and third-party assets reused with attribution.

## Built during the hackathon (original work)

All of `src/`, `eval/`, `ui/`, `tests/` — written from scratch in this repo (see
git history):

- **Module contract** (`src/contract.py`) — the uniform `ModuleResult` /
  `StudyContext` / `PanelOutput` schema and its invariant enforcement.
- **Registry** (`src/registry.py`) — module catalog + calibration source of truth.
- **DICOM pipeline** (`src/dicom_io.py`, `src/windowing.py`) — HU conversion,
  position-based slice ordering, brain/subdural/bone windows.
- **ICH module wrapper** (`src/modules/ich.py`, `ich_model.py`) — inference
  harness + study-level aggregation + Grad-CAM overlays around a reused checkpoint
  (the checkpoint itself is reused — see below).
- **Midline-shift detector** (`src/modules/midline.py`) — an original classical-CV
  algorithm (skull-symmetry reference via PCA + deep-CSF displacement). No ML.
- **Orchestrator + router** (`src/orchestrator.py`, `src/router.py`) — study-type
  routing, unconditional parallel fan-out, failure isolation, superset-on-ambiguous.
- **Claude reconciliation layer** (`src/claude_layer.py`) — downstream-only,
  structured-output verification with a deterministic fallback.
- **Deterministic policy** (`src/policy.py`) — the audited page/reorder split.
- **Eval harness** (`eval/`) — ROC/AUC metrics, router accuracy, calibration.
- **Web UI** (`ui/`) — FastAPI + a single HTML page.

## Reused with attribution (NOT our work)

- **ICH model weights** — a public **RSNA-2019 Intracranial Hemorrhage** solution
  checkpoint (reference: VinBigData CNN stage). We wrap it; we did **not** train it.
  See `scripts/fetch_ich_weights.md`. Non-commercial research terms apply.
- **timm backbone architecture** — EfficientNet via the `timm` library (Apache-2.0).
- **Datasets** — RSNA-2019 ICH and **CQ500** (CC BY-NC-SA), non-commercial only.
  No imaging data or pixel artifacts are committed (`data/`, `artifacts/` gitignored).
- **Claude (Opus 4.8)** — Anthropic API, used as the downstream reconciliation LLM.

## Model inventory

- **1 open-weight detection model**: the reused ICH CNN checkpoint.
- **1 classical-CV detector**: midline shift (no weights).
- **1 hosted LLM**: Claude, for reconciliation/ranking (not open-weight).

## Honesty guarantees enforced in code

- The ICH module runs in a loudly-flagged **untrained** mode until real weights are
  present, and such results are forced **uncalibrated** — never presented as real.
- All reported metrics are **external (CQ500)**; the demo eval cohort is labeled
  `SYNTHETIC-DEMO` and never claimed as CQ500.
- A **non-diagnostic banner** is shown in the UI; the system only re-prioritizes and
  never removes a study from the reading queue.
