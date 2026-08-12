"""Business impact model — ties model probabilities to money.

Two things matter in controlled credit scoring, and neither is "AUC":
  1. **Expected Loss (EL)** of the approved book:  EL = Σ PD · LGD · EAD
  2. The **trade-off** between declining good customers (false positives, lost
     revenue / opportunity cost) and approving future defaulters (false negatives,
     which realise as credit loss).

This module turns scores into KPIs and finds the cost-optimal decision threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import get_config


def _business_params():
    cfg = get_config().business
    return float(cfg.lgd), float(cfg.ead), float(cfg.cost_false_positive)


def expected_loss(pd_scores: np.ndarray, lgd: float | None = None, ead: float | None = None) -> np.ndarray:
    lgd = lgd if lgd is not None else _business_params()[0]
    ead = ead if ead is not None else _business_params()[1]
    return np.asarray(pd_scores, dtype=float) * lgd * ead


def portfolio_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    lgd: float | None = None,
    ead: float | None = None,
    cost_fp: float | None = None,
) -> dict:
    """KPIs at a given PD threshold (approve if PD < threshold).

    Two complementary lenses on cost:
      * ``expected_loss_total`` — *forward-looking* credit risk of the approved
        book: Σ PD·LGD·EAD (no realised labels needed; usable at scoring time).
      * ``total_cost`` — *realised* cost used to tune the threshold:
        realised losses from defaulters we approved (FN · LGD · EAD) plus the
        opportunity cost of good customers we declined (FP · cost_false_positive).
    """
    lgd, ead, cost_fp = _resolve(lgd, ead, cost_fp)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    approved = y_score < threshold  # low-risk applicants are approved

    approval_rate = float(approved.mean())
    # Forward-looking expected credit loss of the approved book.
    el_approved = float(expected_loss(y_score[approved], lgd, ead).sum())
    el_per_approved = el_approved / max(approved.sum(), 1)

    false_negatives = int(((y_true == 1) & approved).sum())      # defaulters we approved
    false_positives = int(((y_true == 0) & ~approved).sum())     # good customers declined
    realised_loss = false_negatives * lgd * ead
    opportunity_cost = false_positives * cost_fp
    total_cost = realised_loss + opportunity_cost

    return {
        "threshold": float(threshold),
        "approval_rate": approval_rate,
        "approved_count": int(approved.sum()),
        "declined_count": int((~approved).sum()),
        "expected_loss_total": el_approved,            # forward-looking (PD-based)
        "expected_loss_per_approved": el_per_approved,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "realised_loss": float(realised_loss),         # FN * LGD * EAD
        "opportunity_cost": float(opportunity_cost),   # FP * cost_false_positive
        "total_cost": float(total_cost),               # the quantity we minimise
    }


def threshold_curve(
    y_true: np.ndarray, y_score: np.ndarray, points: int = 101
) -> pd.DataFrame:
    """Approval rate vs expected loss across thresholds — for the dashboard."""
    rows = [portfolio_metrics(y_true, y_score, t) for t in np.linspace(0.01, 0.99, points)]
    return pd.DataFrame(rows)


def optimal_threshold(
    y_true: np.ndarray, y_score: np.ndarray, grid: int = 200
) -> tuple[float, dict]:
    """Find the PD threshold that minimises total business cost."""
    best_t, best = 0.5, None
    for t in np.linspace(0.01, 0.99, grid):
        m = portfolio_metrics(y_true, y_score, t)
        if best is None or m["total_cost"] < best["total_cost"]:
            best, best_t = m, t
    return float(best_t), best


def _resolve(lgd, ead, cost_fp):
    d_lgd, d_ead, d_fp = _business_params()
    return (
        lgd if lgd is not None else d_lgd,
        ead if ead is not None else d_ead,
        cost_fp if cost_fp is not None else d_fp,
    )
