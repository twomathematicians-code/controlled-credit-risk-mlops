"""Offline synthetic fallback (schema-matched to the Home Credit canonical schema).

Used only when no network / HF is unavailable (e.g. air-gapped dev or as a
CI fallback). It produces the *same columns* as ``data.huggingface`` with a
known latent signal so every downstream stage still runs, but the **default**
data source is Hugging Face — see ``data.huggingface``.

Run as a module: ``python -m credit_risk.data.synthetic``
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ..config import get_config, path
from ..utils.io import write_parquet
from ..utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_RATE = 0.033  # matches the real Home Credit default rate

# Categorical levels (mirroring the real Home Credit values where sensible).
SEX_LEVELS = ["F", "M"]
INCOME_TYPE_LEVELS = [
    "EMPLOYED", "PRIVATE_SECTOR_EMPLOYEE", "RETIRED_PENSIONER",
    "SALARIED_GOVT", "SELFEMPLOYED", "OTHER",
]
FAMILY_STATUS_LEVELS = ["SINGLE", "MARRIED", "DIVORCED", "WIDOWED", "LIVING_WITH_PARTNER"]
EMPLOYMENT_DURATION_LEVELS = ["LESS_ONE", "MORE_ONE", "MORE_FIVE"]
EDUCATION_LEVELS = ["level_1", "level_2", "level_3", "level_4", "level_5", "level_6"]


def _calibrate_intercept(lp: np.ndarray, target_rate: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = (1 / (1 + np.exp(-(lp + mid)))).mean()
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def generate(
    n_samples: int = 60000,
    seed: int | None = None,
    default_rate: float = DEFAULT_RATE,
) -> pd.DataFrame:
    cfg = get_config()
    seed = int(cfg.random_seed if seed is None else seed)
    rng = np.random.default_rng(seed)
    n = int(n_samples)

    age = np.clip(rng.normal(45, 12, n), 21, 85)
    income = np.clip(np.exp(rng.normal(np.log(50000), 0.5, n)), 2000, 200000)
    credit_amount = np.clip(np.exp(rng.normal(np.log(70000), 0.6, n)), 2000, 400000)
    total_debt = np.clip(income * rng.uniform(0.0, 0.6, n), 0, None)
    current_debt = np.clip(total_debt * rng.uniform(0.2, 0.9, n), 0, None)
    num_active_credits = rng.poisson(0.6, n)
    num_credit_inquiries = rng.poisson(1.5, n)
    recent_applications = rng.poisson(1.0, n)
    max_dpd_12m = rng.poisson(0.1, n)            # mostly 0
    num_installments = rng.poisson(2.0, n)

    sex = rng.choice(SEX_LEVELS, size=n)
    income_type = rng.choice(INCOME_TYPE_LEVELS, size=n, p=[0.30, 0.25, 0.15, 0.12, 0.10, 0.08])
    family_status = rng.choice(FAMILY_STATUS_LEVELS, size=n, p=[0.35, 0.40, 0.10, 0.05, 0.10])
    employment_duration = rng.choice(EMPLOYMENT_DURATION_LEVELS, size=n, p=[0.20, 0.45, 0.35])
    education = rng.choice(EDUCATION_LEVELS, size=n)

    # --- Latent risk (the DGP the model must rediscover) ---
    dti = total_debt / np.maximum(income, 1.0)
    emp_effect = np.where(employment_duration == "LESS_ONE", 0.6,
                  np.where(employment_duration == "MORE_FIVE", -0.3, 0.0))
    it_effect = np.where(np.isin(income_type, ["RETIRED_PENSIONER", "SELFEMPLOYED", "OTHER"]), 0.3, 0.0)
    latent = (
        1.2 * np.log1p(max_dpd_12m)
        + 0.9 * np.clip(dti, 0, 2)
        + 0.5 * np.log1p(num_credit_inquiries)
        + 0.3 * np.log1p(num_installments)
        - 0.6 * np.log(income / 50000.0)
        - 0.03 * (age - 45)
        + emp_effect
        + it_effect
    )
    offset = _calibrate_intercept(latent, default_rate, seed + 1)
    pd_proba = 1 / (1 + np.exp(-(latent + offset)))
    default_flag = rng.binomial(1, pd_proba).astype(int)

    df = pd.DataFrame(
        {
            "case_id": np.arange(1, n + 1, dtype=int),
            "age": age,
            "income": income,
            "credit_amount": credit_amount,
            "total_debt": total_debt,
            "current_debt": current_debt,
            "num_active_credits": num_active_credits,
            "num_credit_inquiries": num_credit_inquiries,
            "recent_applications": recent_applications,
            "max_dpd_12m": max_dpd_12m,
            "num_installments": num_installments,
            "sex": sex,
            "education": education,
            "income_type": income_type,
            "family_status": family_status,
            "employment_duration": employment_duration,
            "default_flag": default_flag,
        }
    )
    logger.info(
        "Generated %d synthetic applicants | default_rate=%.3f (target %.3f)",
        n, df["default_flag"].mean(), default_rate,
    )
    return df


def main() -> None:  # pragma: no cover - CLI entry point
    df = generate()
    out = path("data.raw_path")
    write_parquet(df, out)
    logger.info("Wrote raw data -> %s", out)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
