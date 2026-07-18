"""Phase 6: metrics correctness + router-eval accuracy on the labeled cases."""

from __future__ import annotations

import numpy as np

from eval.metrics import roc_curve, sens_spec_at
from eval.router_eval import run as run_router_eval


def test_auc_perfect_separation():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert roc_curve(scores, labels).auc == 1.0


def test_auc_random_is_half_ish():
    rng = np.random.default_rng(0)
    scores = rng.random(2000)
    labels = rng.integers(0, 2, 2000)
    assert 0.4 < roc_curve(scores, labels).auc < 0.6


def test_sens_spec_at_threshold():
    scores = np.array([0.2, 0.4, 0.6, 0.8])
    labels = np.array([0, 0, 1, 1])
    sens, spec = sens_spec_at(scores, labels, 0.5)
    assert sens == 1.0 and spec == 1.0


def test_auc_monotone_with_signal():
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.4).astype(int)
    strong = np.where(y == 1, rng.normal(0.8, 0.1, 500), rng.normal(0.2, 0.1, 500))
    weak = np.where(y == 1, rng.normal(0.55, 0.25, 500), rng.normal(0.45, 0.25, 500))
    assert roc_curve(strong, y).auc > roc_curve(weak, y).auc


def test_precision_at_recall():
    import numpy as np

    from eval.metrics import precision_at_recall

    s = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0, 0, 1, 1])
    prec, _ = precision_at_recall(s, y, 1.0)
    assert prec == 1.0


def test_verification_eval_demo_shows_improvement():
    """Claude's down-weights should raise precision at fixed recall on the demo."""
    from eval.verification_eval import demo

    res = demo(recall=0.9)
    assert res["precision_with_claude"] > res["precision_detector"]
    assert res["delta"] > 0


def test_router_eval_accuracy_reasonable():
    res = run_router_eval()
    assert res["n"] >= 20
    # The keyword router should get the clear cases right.
    assert res["accuracy"] >= 0.85
