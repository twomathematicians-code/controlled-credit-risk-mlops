"""Feature engineering tests."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from credit_risk.features.engineering import (
    INDICATOR_SPECS,
    RATIO_SPECS,
    RatioAdder,
    build_pipeline,
    transform_for_explanation,
)


def test_ratio_adder_creates_engineered_columns(small_df, feature_cols):
    added = RatioAdder().fit_transform(small_df[feature_cols])
    for name in RATIO_SPECS:
        assert name in added.columns
    for name in INDICATOR_SPECS:
        assert name in added.columns
    # inquiry_intensity denominator is (active_credits + 1) -> never inf / NaN.
    assert added["inquiry_intensity"].replace([np.inf, -np.inf], np.nan).isna().sum() == 0
    # dpd_flag is binary.
    assert set(added["dpd_flag"].unique()).issubset({0.0, 1.0})


def test_pipeline_fit_predict(small_df, feature_cols, target_col):
    pipe = build_pipeline(LogisticRegression(max_iter=300))
    pipe.fit(small_df[feature_cols], small_df[target_col])
    proba = pipe.predict_proba(small_df[feature_cols])[:, 1]
    assert proba.shape == (len(small_df),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_pipeline_learns_signal(small_df, feature_cols, target_col):
    pipe = build_pipeline(LogisticRegression(max_iter=300))
    pipe.fit(small_df[feature_cols], small_df[target_col])
    auc = roc_auc_score(small_df[target_col], pipe.predict_proba(small_df[feature_cols])[:, 1])
    # On the data it was fit on, signal must be clearly better than chance.
    assert auc > 0.65


def test_transform_for_explanation_shape(fitted_pipeline, small_df, feature_cols):
    X = small_df[feature_cols].head(10)
    Xt, names = transform_for_explanation(fitted_pipeline, X)
    assert Xt.shape[0] == 10
    assert Xt.shape[1] == len(names)
