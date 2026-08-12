"""Shared test fixtures. Tests use small synthetic data so the suite is fast."""
from __future__ import annotations

import pytest
from sklearn.linear_model import LogisticRegression

from credit_risk.config import get_config
from credit_risk.data.synthetic import generate
from credit_risk.features.engineering import build_pipeline


@pytest.fixture(scope="session")
def small_df():
    return generate(n_samples=800, seed=7, default_rate=0.12, fraud_rate=0.02)


@pytest.fixture(scope="session")
def feature_cols():
    cfg = get_config().data
    return list(cfg.numeric_features) + list(cfg.categorical_features)


@pytest.fixture(scope="session")
def target_col():
    return get_config().data.target_column


@pytest.fixture(scope="session")
def fitted_pipeline(small_df, feature_cols, target_col):
    X, y = small_df[feature_cols], small_df[target_col]
    pipe = build_pipeline(LogisticRegression(max_iter=300))
    pipe.fit(X, y)
    return pipe
