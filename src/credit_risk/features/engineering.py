"""Feature engineering pipeline.

Builds a reusable, serialisable scikit-learn pipeline:
    [RatioAdder] -> ColumnTransformer{numeric: impute+scale, cat: impute+onehot}

Engineered ratios (defined in config) capture the strongest credit-risk signals:
debt-to-income, credit utilisation, inquiry intensity and payment burden. The
same pipeline is used at train time and at serve time, guaranteeing that live
requests are transformed identically to training data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import get_config

# (engineered name) -> (numerator column, denominator column)
RATIO_SPECS = {
    "debt_to_income": ("total_debt", "annual_income"),
    "credit_utilization": ("credit_card_balance", "credit_limit"),
    "inquiries_per_account": ("num_credit_inquiries_12m", "num_open_accounts"),
    "payment_burden": ("missed_payments_12m", None),  # denominator constant (12 months)
}


class RatioAdder(BaseEstimator, TransformerMixin):
    """Add engineered ratio features to a DataFrame, guarding divide-by-zero."""

    def __init__(self, specs: dict[str, tuple] | None = None):
        self.specs = specs or RATIO_SPECS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        for name, (num, den) in self.specs.items():
            numerator = df[num].astype(float)
            if den is None:
                df[name] = numerator / 12.0
            else:
                denominator = df[den].replace(0, np.nan).astype(float)
                df[name] = (numerator / denominator).replace([np.inf, -np.inf], np.nan)
        return df


def _columns():
    cfg = get_config().features
    numeric = list(cfg.numeric_features)
    categorical = list(cfg.categorical_features)
    engineered = [n for n in cfg.engineered if n in RATIO_SPECS]
    return numeric, categorical, engineered


def build_preprocessor() -> Pipeline:
    """Return the feature-engineering + preprocessing pipeline (no estimator)."""
    numeric, categorical, engineered = _columns()

    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric + engineered),
            ("cat", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("ratios", RatioAdder()), ("preprocess", pre)])


def build_pipeline(estimator) -> Pipeline:
    """Full train-time pipeline: preprocessing + the supplied estimator."""
    return Pipeline(steps=[("features", build_preprocessor()), ("model", estimator)])


def transform_for_explanation(pipeline: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return the transformed feature matrix + names (for SHAP on the inner model).

    Assumes ``pipeline`` has ``features`` (preprocessor) and ``model`` steps.
    """
    pre: Pipeline = pipeline.named_steps["features"]
    transformed = pre.transform(X)
    names = _output_feature_names(pre)
    return transformed, names


def _output_feature_names(preprocessor: Pipeline) -> list[str]:
    ct: ColumnTransformer = preprocessor.named_steps["preprocess"]
    try:
        names = list(ct.get_feature_names_out())
    except Exception:
        numeric, categorical, engineered = _columns()
        names = numeric + engineered + [f"cat__{c}" for c in categorical]
    return names
