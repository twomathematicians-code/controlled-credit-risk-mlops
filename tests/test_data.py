"""Data layer tests: synthetic generation + ingestion schema validation."""
from __future__ import annotations

import pandas as pd
import pytest

from credit_risk.config import get_config
from credit_risk.data import ingestion, synthetic


def test_generate_shape_and_columns(small_df):
    cfg = get_config().data
    expected = set(cfg.numeric_features) | set(cfg.categorical_features) | {cfg.id_column, cfg.target_column, "is_fraud"}
    assert expected.issubset(small_df.columns)
    assert len(small_df) == 800


def test_generate_target_rate_close_to_target():
    df = synthetic.generate(n_samples=4000, seed=3, default_rate=0.12, fraud_rate=0.02)
    rate = df[get_config().data.target_column].mean()
    # Loose bound — stochastic, but calibration should land near target.
    assert 0.07 <= rate <= 0.17


def test_generate_has_fraud_subset(small_df):
    assert small_df["is_fraud"].sum() > 0
    # Fraudsters should default at a higher rate than the general population.
    target = get_config().data.target_column
    assert small_df.loc[small_df.is_fraud == 1, target].mean() >= small_df[target].mean()


def test_validate_schema_rejects_missing_columns():
    df = synthetic.generate(n_samples=50, seed=1)
    df = df.drop(columns=["annual_income"])
    with pytest.raises(ValueError, match="missing columns"):
        ingestion.validate_schema(df)


def test_validate_schema_rejects_duplicate_ids():
    df = synthetic.generate(n_samples=50, seed=1)
    df.loc[1, get_config().data.id_column] = df.loc[0, get_config().data.id_column]
    with pytest.raises(ValueError, match="Duplicate"):
        ingestion.validate_schema(df)


def test_validate_schema_coerces_and_fills(small_df):
    out = ingestion.validate_schema(small_df.copy())
    cfg = get_config().data
    for col in cfg.numeric_features:
        assert pd.api.types.is_numeric_dtype(out[col])
    # Categoricals get a sentinel policy for nulls.
    for col in cfg.categorical_features:
        assert out[col].isna().sum() == 0
