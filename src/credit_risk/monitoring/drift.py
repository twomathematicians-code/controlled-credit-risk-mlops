"""Drift detection — Population Stability Index (PSI) + Kolmogorov–Smirnov.

The drift reference is the frozen training distribution (see ingestion). On each
monitoring run we compare the *current* feature distribution (e.g. a batch of
live applicants) against that reference and emit a pass/warn/fail verdict per
feature plus an overall verdict.

  PSI < 0.10  -> no significant drift          (pass)
  0.10–0.25   -> slight drift, investigate     (warn)
  > 0.25      -> significant drift, retrain     (fail)

Thresholds live in ``config.yaml`` so they are governed, not hard-coded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from ..config import get_config, path
from ..utils.logging import get_logger

logger = get_logger(__name__)
_EPS = 1e-4


def status_from_psi(psi: float, warn: float | None = None, fail: float | None = None) -> str:
    cfg = get_config().monitoring.drift
    warn = cfg.psi_warn if warn is None else warn
    fail = cfg.psi_fail if fail is None else fail
    if psi >= fail:
        return "fail"
    if psi >= warn:
        return "warn"
    return "pass"


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI for a numeric variable using reference quantile bins."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    # Drop NaNs for bin construction.
    ref_clean = reference[~np.isnan(reference)]
    if len(ref_clean) < bins:
        return 0.0
    edges = np.unique(np.percentile(ref_clean, np.linspace(0, 100, bins + 1)))
    if len(edges) < 3:  # near-constant feature — PSI is not meaningful
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = np.histogram(reference, bins=edges)[0].astype(float)
    cur_counts = np.histogram(current, bins=edges)[0].astype(float)

    ref_pct = ref_counts / max(ref_counts.sum(), 1) + _EPS
    cur_pct = cur_counts / max(cur_counts.sum(), 1) + _EPS
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    """PSI for a categorical variable based on category frequencies."""
    ref_counts = reference.astype(str).value_counts(normalize=True)
    cur_counts = current.astype(str).value_counts(normalize=True)
    categories = ref_counts.index.union(cur_counts.index)
    ref_pct = ref_counts.reindex(categories, fill_value=0.0).to_numpy() + _EPS
    cur_pct = cur_counts.reindex(categories, fill_value=0.0).to_numpy() + _EPS
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _ks(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) < 2 or len(cur) < 2:
        return 0.0, 1.0
    res = ks_2samp(ref, cur)
    return float(res.statistic), float(res.pvalue)


def feature_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> dict:
    warn, fail = _thresholds()
    rows = {}
    for col in numeric_cols:
        if col not in reference.columns or col not in current.columns:
            continue
        psi = compute_psi(reference[col].to_numpy(), current[col].to_numpy())
        ks_stat, ks_p = _ks(reference[col].to_numpy(), current[col].to_numpy())
        rows[col] = {
            "psi": round(psi, 4),
            "ks_statistic": round(ks_stat, 4),
            "ks_pvalue": round(ks_p, 4),
            "status": status_from_psi(psi, warn, fail),
        }
    for col in categorical_cols:
        if col not in reference.columns or col not in current.columns:
            continue
        psi = compute_psi_categorical(reference[col], current[col])
        rows[col] = {"psi": round(psi, 4), "status": status_from_psi(psi, warn, fail)}
    return rows


def score_drift(reference_scores: np.ndarray, current_scores: np.ndarray) -> dict:
    warn, fail = _thresholds()
    psi = compute_psi(np.asarray(reference_scores), np.asarray(current_scores))
    ks_stat, ks_p = _ks(np.asarray(reference_scores), np.asarray(current_scores))
    return {
        "psi": round(psi, 4),
        "ks_statistic": round(ks_stat, 4),
        "ks_pvalue": round(ks_p, 4),
        "status": status_from_psi(psi, warn, fail),
    }


def overall_status(report: dict) -> str:
    statuses = [v.get("status", "pass") for v in report.values()]
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


def evaluate(current: pd.DataFrame, scores: np.ndarray | None = None) -> dict:
    """Run a full drift check vs the frozen reference. Returns the consolidated report."""
    cfg = get_config()
    reference = pd.read_parquet(path("data.drift_reference_path"))
    numeric_cols = list(cfg.data.numeric_features)
    categorical_cols = list(cfg.data.categorical_features)
    numeric_cols = [c for c in numeric_cols if c in reference.columns and c in current.columns]
    categorical_cols = [c for c in categorical_cols if c in reference.columns and c in current.columns]

    feature_report = feature_drift_report(reference, current, numeric_cols, categorical_cols)
    verdict = overall_status(feature_report)
    result = {"features": feature_report, "overall_status": verdict}

    if scores is not None and cfg.data.target_column in reference.columns:
        # Use the reference *label* rate only for context; score drift uses the
        # reference applicants' realised PD if recorded, else skipped.
        result["score_drift"] = score_drift(reference[cfg.data.target_column].to_numpy(), scores)
    logger.info("Drift overall_status=%s across %d features", verdict, len(feature_report))
    return result


def _thresholds() -> tuple[float, float]:
    cfg = get_config().monitoring.drift
    return float(cfg.psi_warn), float(cfg.psi_fail)
