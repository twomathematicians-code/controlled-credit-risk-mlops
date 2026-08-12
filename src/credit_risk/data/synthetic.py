"""Synthetic credit-applicant data generator.

Why synthetic?
  * Fully self-contained repo (no dataset download / licensing) — ideal for CI.
  * We *control the data-generating process*, so there is genuine learnable signal
    (the latent risk score) and a known target default rate.

The generator:
  1. Draws raw applicant features from sensible, lightly-correlated distributions.
  2. Computes a latent risk score from realistic ratios (DTI, utilisation, missed
     payments, employment tenure, income, employment status).
  3. Calibrates the intercept so the realised default rate ~= ``default_rate``.
  4. Injects an ``is_fraud`` subset (income materially overstated) so the dataset
     also supports a fraud-scoring framing — these cases default more often.

Run as a module: ``python -m credit_risk.data.synthetic``.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ..config import get_config, path
from ..utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Reproducible sampling helpers
# ---------------------------------------------------------------------------
def _lognormal(rng, mean, sd, size, low=None, high=None):
    """Sample from a lognormal whose underlying normal has given mean/sd of log."""
    x = np.exp(rng.normal(np.log(mean), sd, size=size))
    if low is not None or high is not None:
        x = np.clip(x, low if low is not None else x.min(), high if high is not None else x.max())
    return x


def _calibrate_intercept(lp: np.ndarray, target_rate: float, seed: int) -> float:
    """Find intercept offset so mean(sigmoid(lp + offset)) ~= target_rate."""
    rng = np.random.default_rng(seed)
    lo, hi = -10.0, 10.0
    target = target_rate
    for _ in range(60):  # bisection — sigmoid monotonic in offset
        mid = (lo + hi) / 2
        rate = (1 / (1 + np.exp(-(lp + mid)))).mean()
        if rate < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------
def generate(
    n_samples: int | None = None,
    seed: int | None = None,
    default_rate: float | None = None,
    fraud_rate: float | None = None,
) -> pd.DataFrame:
    """Generate a synthetic applicant DataFrame with a ``default_flag`` target."""
    cfg = get_config().data
    n_samples = int(cfg.n_samples if n_samples is None else n_samples)
    seed = int(get_config().random_seed if seed is None else seed)
    default_rate = float(cfg.default_rate if default_rate is None else default_rate)
    fraud_rate = float(cfg.fraud_rate if fraud_rate is None else fraud_rate)

    rng = np.random.default_rng(seed)
    n = n_samples

    age = np.clip(rng.normal(42, 12, n).round(), 18, 85).astype(int)
    true_income = _lognormal(rng, mean=48000, sd=0.45, size=n, low=8000, high=350000)
    months_employed = np.clip(rng.exponential(scale=70, size=n), 0, 360).round().astype(int)
    num_open_accounts = (rng.poisson(5, n) + 1).astype(int)
    inquiries = rng.poisson(1.2, n).astype(int)
    missed_payments = rng.poisson(0.45, n).astype(int)
    credit_limit = np.clip(true_income * rng.uniform(0.15, 0.6, n), 500, 120000)
    credit_card_balance = np.clip(credit_limit * rng.uniform(0.0, 0.95, n), 0, None)
    total_debt = np.clip(
        true_income * rng.uniform(0.0, 0.85, n) + num_open_accounts * rng.uniform(800, 4000, n),
        0,
        None,
    )

    employment_status = rng.choice(
        ["Employed", "Self-Employed", "Unemployed", "Retired"],
        size=n,
        p=[0.62, 0.18, 0.12, 0.08],
    )
    home_ownership = rng.choice(
        ["Rent", "Mortgage", "Own", "Other"], size=n, p=[0.38, 0.42, 0.17, 0.03]
    )
    loan_purpose = rng.choice(
        ["debt_consolidation", "home_improvement", "major_purchase", "credit_card", "other"],
        size=n,
        p=[0.34, 0.20, 0.18, 0.18, 0.10],
    )
    region = rng.choice(
        ["Capital", "North", "South", "West", "East"], size=n, p=[0.30, 0.25, 0.20, 0.15, 0.10]
    )

    # --- Fraud subset: income materially overstated (reported >> true) ----------
    is_fraud = rng.random(n) < fraud_rate
    reported_income = true_income.copy()
    reported_income[is_fraud] = reported_income[is_fraud] * rng.uniform(2.5, 4.0, is_fraud.sum())

    # --- Latent risk from TRUE quantities (the DGP the model must rediscover) ---
    dti = total_debt / np.maximum(reported_income, 1.0)
    util = credit_card_balance / np.maximum(credit_limit, 1.0)
    unemp = (employment_status == "Unemployed").astype(float)

    latent = (
        1.6 * np.log1p(missed_payments)
        + 1.2 * util
        + 0.9 * np.clip(dti, 0, 2)
        + 0.35 * np.log1p(inquiries)
        - 0.30 * np.log1p(months_employed)
        - 0.55 * np.log(true_income / 48000.0)
        - 0.04 * (age - 42)
        + 0.9 * unemp
    )
    # Fraudsters are riskier in reality (their true income is far lower).
    latent = latent + is_fraud.astype(float) * 1.1

    offset = _calibrate_intercept(latent, default_rate, seed + 1)
    pd_proba = 1 / (1 + np.exp(-(latent + offset)))
    default_flag = rng.binomial(1, pd_proba).astype(int)

    df = pd.DataFrame(
        {
            "application_id": np.arange(1, n + 1, dtype=int),
            "age": age,
            "annual_income": reported_income.round(2),
            "months_employed": months_employed,
            "num_open_accounts": num_open_accounts,
            "num_credit_inquiries_12m": inquiries,
            "total_debt": total_debt.round(2),
            "credit_limit": credit_limit.round(2),
            "missed_payments_12m": missed_payments,
            "credit_card_balance": credit_card_balance.round(2),
            "employment_status": employment_status,
            "home_ownership": home_ownership,
            "loan_purpose": loan_purpose,
            "region": region,
            "is_fraud": is_fraud.astype(int),
            "default_flag": default_flag,
        }
    )
    logger.info(
        "Generated %d applicants | default_rate=%.3f (target %.3f) | fraud_rate=%.3f",
        n,
        df["default_flag"].mean(),
        default_rate,
        df["is_fraud"].mean(),
    )
    return df


def main() -> None:  # pragma: no cover - CLI entry point
    from ..utils.io import write_parquet

    df = generate()
    out = path("data.raw_path")
    write_parquet(df, out)
    logger.info("Wrote raw data -> %s", out)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
