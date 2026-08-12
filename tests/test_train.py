"""Training utilities tests (light — full training is exercised via the CLI)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from credit_risk.models import train


def test_build_estimators_returns_configured_models():
    estimators = train.build_estimators()
    assert set(estimators) == {"logreg", "gbdt", "xgboost"}
    assert isinstance(estimators["logreg"], LogisticRegression)
    assert isinstance(estimators["gbdt"], GradientBoostingClassifier)
    from xgboost import XGBClassifier

    assert isinstance(estimators["xgboost"], XGBClassifier)


def test_bootstrap_auc_ci_ordering():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    scores = rng.random(300)
    point, lo, hi = train.bootstrap_auc_ci(y, scores, n_boot=50, seed=0)
    assert lo <= point <= hi
    assert lo < hi


def test_cv_scores_shape(small_df, feature_cols, target_col):
    scores = train._cv_scores(
        LogisticRegression(max_iter=300), small_df[feature_cols], small_df[target_col], seed=1, folds=3
    )
    assert "roc_auc" in scores
    mean, std = scores["roc_auc"]
    assert 0.0 <= mean <= 1.0
    assert std >= 0
