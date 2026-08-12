"""Hugging Face data loader — Home Credit Default Risk.

Primary data source. Downloads the processed Home Credit parquet from the Hub,
maps the raw (cryptic) column names to a clean canonical schema, coerces types,
and reproducibly samples a stratified subset for fast iteration.

Dataset: ``deburky/home-credit-credit-risk-model-stability`` — 522,596 real loan
applications, binary ``target`` (~3.3% default). Public, no auth required.

Run as a module: ``python -m credit_risk.data.huggingface``
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ..config import get_config, path
from ..utils.io import write_parquet
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Raw Home Credit column -> clean canonical name
NUMERIC_MAP = {
    "age": "age",
    "mainoccupationinc_384A": "income",
    "credamount_770A": "credit_amount",
    "totaldebt_9A": "total_debt",
    "currdebt_22A": "current_debt",
    "numactivecreds_622L": "num_active_credits",
    "numberofqueries_373L": "num_credit_inquiries",
    "applicationscnt_1086L": "recent_applications",
    "maxdpdlast12m_727P": "max_dpd_12m",
    "numinstls_657L": "num_installments",
}
CATEGORICAL_MAP = {
    "sex_738L": "sex",
    "education_927M": "education",
    "incometype_1044T": "income_type",
    "familystate_447L": "family_status",
    "empl_employedtotal_800L": "employment_duration",
}
TARGET_MAP = {"target": "default_flag"}
ID_COL = "case_id"


def _download() -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    cfg = get_config().data.huggingface
    local = hf_hub_download(repo_id=cfg.repo_id, filename=cfg.filename, repo_type="dataset")
    logger.info("Downloaded %s -> %s", cfg.repo_id, local)
    return pd.read_parquet(local)


def _map_to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    mapping = {**NUMERIC_MAP, **CATEGORICAL_MAP, **TARGET_MAP}
    keep = [ID_COL] + [c for c in mapping if c in raw.columns]
    df = raw[keep].rename(columns=mapping).copy()

    for col in NUMERIC_MAP.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_MAP.values():
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def _stratified_sample(df: pd.DataFrame, sample_size: int | None, seed: int) -> pd.DataFrame:
    if sample_size is None or sample_size <= 0 or sample_size >= len(df):
        return df.reset_index(drop=True)
    target = get_config().data.target_column
    rng = np.random.default_rng(seed)
    frac = sample_size / len(df)
    parts = []
    for _, sub in df.groupby(target):
        n = max(1, int(round(len(sub) * frac)))
        parts.append(sub.sample(n=min(n, len(sub)), random_state=int(rng.integers(1e9))))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load(sample_size: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """Download + clean + map + sample the Home Credit dataset to canonical schema."""
    cfg = get_config()
    seed = int(cfg.random_seed if seed is None else seed)
    if sample_size is None:
        sample_size = cfg.data.huggingface.get("sample_size", None)
        sample_size = int(sample_size) if sample_size else None

    raw = _download()
    df = _map_to_canonical(raw)
    df = _stratified_sample(df, sample_size, seed)

    target = cfg.data.target_column
    logger.info(
        "Home Credit loaded: %d rows | default_rate=%.3f | numeric=%d categorical=%d",
        len(df),
        df[target].mean(),
        len(NUMERIC_MAP),
        len(CATEGORICAL_MAP),
    )
    return df


def main() -> None:  # pragma: no cover - CLI entry point
    df = load()
    out = path("data.raw_path")
    write_parquet(df, out)
    logger.info("Wrote raw data -> %s", out)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
