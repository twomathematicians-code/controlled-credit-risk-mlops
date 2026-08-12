"""Drift detection tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.monitoring import drift


def test_psi_zero_for_identical_distribution():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 2000)
    psi = drift.compute_psi(x, x)
    assert psi < 0.02


def test_psi_positive_for_shifted_distribution():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 2000)
    cur = rng.normal(1.0, 1.5, 2000)  # clearly shifted
    psi = drift.compute_psi(ref, cur)
    assert psi > 0.25


def test_status_thresholds():
    assert drift.status_from_psi(0.01) == "pass"
    assert drift.status_from_psi(0.15) == "warn"
    assert drift.status_from_psi(0.40) == "fail"


def test_feature_drift_report_numeric_and_categorical():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 500), "cat": rng.choice(["x", "y"], 500)})
    cur = pd.DataFrame({"a": rng.normal(0.8, 1.2, 500), "cat": rng.choice(["x", "z"], 500)})
    report = drift.feature_drift_report(ref, cur, numeric_cols=["a"], categorical_cols=["cat"])
    assert report["a"]["status"] == "fail"
    assert report["cat"]["psi"] >= 0.0


def test_overall_status_aggregation():
    report = {"f1": {"status": "pass"}, "f2": {"status": "warn"}}
    assert drift.overall_status(report) == "warn"
    report["f2"]["status"] = "fail"
    assert drift.overall_status(report) == "fail"
