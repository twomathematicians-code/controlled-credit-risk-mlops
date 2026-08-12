"""Data ingestion: raw -> validated -> processed splits + frozen drift reference.

Responsibilities (a "controlled" ingestion contract):
  * Schema validation — required columns present, types coercible, no dup ids.
  * Null policy — categoricals filled with a sentinel; numerics left NaN for the
    imputer (so the policy is explicit, not implicit).
  * Stratified train/test split (the test set mimics future, unseen applicants).
  * Freeze a *drift reference* snapshot (training distribution) so monitoring has
    a stable baseline to compare live traffic against.
"""
from __future__ import annotations

import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import get_config, path
from ..utils.io import read_parquet, write_parquet
from ..utils.logging import get_logger

logger = get_logger(__name__)


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the raw schema and coerce types. Raises on contract violation."""
    cfg = get_config().data
    required_numeric = list(cfg.numeric_features)
    required_categorical = list(cfg.categorical_features)
    id_col, target = cfg.id_column, cfg.target_column

    missing = [c for c in required_numeric + required_categorical + [id_col, target] if c not in df.columns]
    if missing:
        raise ValueError(f"Schema validation failed — missing columns: {missing}")

    if df[id_col].duplicated().any():
        raise ValueError(f"Duplicate {id_col} values found in raw data.")

    for col in required_numeric + ([target] if target not in required_numeric else []):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in required_categorical:
        df[col] = df[col].astype("string").fillna("Missing")
    # Categoricals: explicit sentinel for any null (policy = "Missing").
    for col in required_categorical:
        df[col] = df[col].replace("", "Missing").astype("category")
    return df


def apply_null_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Numerics: leave NaN for the pipeline imputer (documented, not silent)."""
    return df


def run() -> dict[str, pd.DataFrame]:
    """Run the full ingestion pipeline and persist outputs. Returns the splits."""
    cfg = get_config()
    data_cfg = cfg.data
    seed = cfg.random_seed

    raw_path = path("data.raw_path")
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}. Run `python -m credit_risk.data.synthetic` first."
        )
    df = read_parquet(raw_path)
    logger.info("Loaded %d raw rows from %s", len(df), raw_path)

    df = validate_schema(df)
    df = apply_null_policy(df)

    feature_cols = list(data_cfg.numeric_features) + list(data_cfg.categorical_features)
    cols = [data_cfg.id_column, *feature_cols, data_cfg.target_column, "is_fraud"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()

    train_df, test_df = train_test_split(
        df, test_size=float(data_cfg.test_size), random_state=seed, stratify=df[data_cfg.target_column]
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    write_parquet(df, path("data.processed_path"))
    write_parquet(train_df, path("data.train_path"))
    write_parquet(test_df, path("data.test_path"))

    # Frozen drift reference: training distribution of scoring features + label.
    ref = train_df[feature_cols + [data_cfg.target_column]].copy()
    write_parquet(ref, path("data.drift_reference_path"))

    logger.info(
        "Ingestion done | processed=%d train=%d test=%d | default_rate train=%.3f test=%.3f",
        len(df),
        len(train_df),
        len(test_df),
        train_df[data_cfg.target_column].mean(),
        test_df[data_cfg.target_column].mean(),
    )
    return {"processed": df, "train": train_df, "test": test_df, "reference": ref}


def main() -> None:  # pragma: no cover
    run()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
