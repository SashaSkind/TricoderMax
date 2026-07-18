"""
Orchestrator — study-type routing + unconditional parallel fan-out with failure
isolation.

Guarantees (see invariants):
  - The panel runs UNCONDITIONALLY: every applicable module runs; there is no
    pre-screen or early exit (invariant 3).
  - Ambiguous study type → the SUPERSET of modules runs (invariant 4).
  - A module that crashes or times out yields status="failed" and never blocks the
    rest of the panel (failure isolation).
  - Routing uses study metadata only, never pixels (invariant 2).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime, timezone

import numpy as np

from src import config
from src.contract import ModuleResult, PanelOutput, StudyContext, StudyType
from src.logging_conf import get_logger
from src.modules.base import Module
from src.modules.factory import build_module
from src.registry import descriptors_for

log = get_logger(__name__)


def route(ctx: StudyContext) -> tuple[list[str], bool]:
    """Pick module ids for a study. Returns (module_ids, superset_run)."""
    superset = ctx.study_type == StudyType.unknown
    descriptors = descriptors_for(ctx.study_type)
    return [d.module_id for d in descriptors], superset


def _run_one(module: Module, volume: np.ndarray, ctx: StudyContext) -> ModuleResult:
    """Run a single module, converting any exception into a failed result."""
    start = time.time()
    try:
        if not module.applies_to(ctx):
            return module.not_applicable_result("module does not apply to study type")
        return module.run(volume, ctx)
    except Exception as e:  # noqa: BLE001 — failure isolation is the whole point
        ms = int((time.time() - start) * 1000)
        log.error("module.exception", accession=ctx.accession, module=module.module_id, err=str(e))
        return module.failed_result(f"{type(e).__name__}: {e}", runtime_ms=ms)


def run_panel(ctx: StudyContext, volume: np.ndarray) -> PanelOutput:
    """Fan the detector panel out over `volume` and assemble a PanelOutput."""
    slog = log.bind(accession=ctx.accession)
    module_ids, superset = route(ctx)
    ctx = ctx.model_copy(update={"routing_ambiguous": superset})
    slog.info("panel.start", modules=module_ids, superset=superset, study_type=ctx.study_type.value)

    started = datetime.now(timezone.utc)
    modules = [build_module(mid) for mid in module_ids]
    results: dict[str, ModuleResult] = {}

    # Parallel fan-out with a per-module wall-clock timeout.
    with ThreadPoolExecutor(max_workers=max(1, len(modules))) as ex:
        futures = {ex.submit(_run_one, m, volume, ctx): m for m in modules}
        for fut, m in list(futures.items()):
            try:
                results[m.module_id] = fut.result(timeout=config.MODULE_TIMEOUT_S)
            except FutureTimeout:
                slog.error("module.timeout", module=m.module_id, timeout_s=config.MODULE_TIMEOUT_S)
                results[m.module_id] = m.failed_result(
                    f"timeout after {config.MODULE_TIMEOUT_S:.0f}s",
                    runtime_ms=int(config.MODULE_TIMEOUT_S * 1000),
                )

    completed = datetime.now(timezone.utc)
    ordered = [results[mid] for mid in module_ids]  # stable order = routing order
    failed = [mid for mid in module_ids if results[mid].status == "failed"]

    panel = PanelOutput(
        study=ctx,
        results=ordered,
        modules_selected=module_ids,
        modules_failed=failed,
        superset_run=superset,
        started_at=started,
        completed_at=completed,
        total_runtime_ms=int((completed - started).total_seconds() * 1000),
    )
    slog.info(
        "panel.done",
        n_positive=len(panel.positive_results),
        n_failed=len(failed),
        runtime_ms=panel.total_runtime_ms,
    )
    return panel
