"""Business impact model tests."""
from __future__ import annotations

import numpy as np

from credit_risk import business


def test_expected_loss_formula():
    el = business.expected_loss(np.array([0.0, 0.5, 1.0]), lgd=0.5, ead=1000)
    np.testing.assert_allclose(el, [0.0, 250.0, 500.0])


def test_portfolio_metrics_monotonic_approval():
    # Higher threshold => approve more => approval_rate non-decreasing with threshold.
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    scores = rng.random(500)
    rates = [business.portfolio_metrics(y, scores, t)["approval_rate"] for t in (0.2, 0.5, 0.8)]
    assert rates[0] <= rates[1] <= rates[2]


def test_optimal_threshold_in_range():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    scores = rng.random(400)
    t, metrics = business.optimal_threshold(y, scores)
    assert 0.0 < t < 1.0
    assert metrics["total_cost"] >= 0
    assert "approval_rate" in metrics


def test_threshold_curve_columns():
    rng = np.random.default_rng(2)
    curve = business.threshold_curve(rng.integers(0, 2, 100), rng.random(100), points=20)
    assert len(curve) == 20
    assert {"threshold", "approval_rate", "expected_loss_total"}.issubset(curve.columns)
