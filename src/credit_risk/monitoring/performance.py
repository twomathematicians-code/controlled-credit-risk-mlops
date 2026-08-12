"""Performance monitoring — realised metrics + governed alert thresholds.

Once outcomes (``default_flag``) become available for a batch of scored
applicants, this module recomputes the headline metrics and flags breaches
against the thresholds defined in ``config.yaml``. Alerts are returned (not
emailed) so callers (CI, a scheduler, the dashboard) can route them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..config import get_config, path
from ..utils.logging import get_logger

logger = get_logger(__name__)


def compute_metrics(y_true, y_score, threshold: float) -> dict:
    """Core metrics at a given PD threshold (predict default if PD >= threshold)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "realised_default_rate": float(y_true.mean()),
        "approval_rate": float((y_score < threshold).mean()),
    }
    return metrics


def reference_default_rate() -> float:
    """Baseline realised default rate from the frozen drift reference."""
    ref = pd.read_parquet(path("data.drift_reference_path"))
    target = get_config().data.target_column
    return float(ref[target].mean()) if target in ref.columns else float("nan")


def evaluate(y_true, y_score, threshold: float | None = None) -> dict:
    """Compute metrics + emit alert list against governed thresholds."""
    cfg = get_config()
    threshold = float(cfg.serving.default_threshold if threshold is None else threshold)
    metrics = compute_metrics(y_true, y_score, threshold)
    perf_cfg = cfg.monitoring.performance

    alerts: list[dict] = []
    if metrics["roc_auc"] < float(perf_cfg.min_roc_auc):
        alerts.append({
            "metric": "roc_auc",
            "value": metrics["roc_auc"],
            "threshold": float(perf_cfg.min_roc_auc),
            "direction": "below_min",
        })
    if metrics["precision"] < float(perf_cfg.min_precision):
        alerts.append({
            "metric": "precision",
            "value": metrics["precision"],
            "threshold": float(perf_cfg.min_precision),
            "direction": "below_min",
        })

    ref_default = reference_default_rate()
    if not np.isnan(ref_default):
        drift = abs(metrics["realised_default_rate"] - ref_default)
        metrics["reference_default_rate"] = ref_default
        metrics["default_rate_drift"] = float(drift)
        if drift > float(perf_cfg.max_default_rate_drift):
            alerts.append({
                "metric": "default_rate_drift",
                "value": float(drift),
                "threshold": float(perf_cfg.max_default_rate_drift),
                "direction": "above_max",
            })

    metrics["alerts"] = alerts
    metrics["status"] = "alert" if alerts else "ok"
    logger.info("Performance status=%s alerts=%d roc_auc=%.4f", metrics["status"], len(alerts), metrics["roc_auc"])
    return metrics
