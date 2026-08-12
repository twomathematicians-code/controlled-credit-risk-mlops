"""Feature engineering pipeline.

Builds a reusable, serialisable scikit-learn pipeline:
    [RatioAdder] -> ColumnTransformer{numeric: impute+scale, cat: impute+onehot}

Engineered features (defined in config) capture the strongest credit-risk
signals from the Home Credit schema: debt/credit-to-income ratios, inquiry
intensity, and a days-past-due indicator. The same pipeline is used at train
and serve time, guaranteeing live requests are transformed identically.
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

# name -> (numerator_col, denominator_col, additive offset on denominator)
# value = numerator / (denominator + offset)
RATIO_SPECS = {
    "debt_to_income": ("total_debt", "income", 0),
    "credit_to_income": ("credit_amount", "income", 0),
    "inquiry_intensity": ("num_credit_inquiries", "num_active_credits", 1),
}
# name -> (column, threshold)  -> value = 1.0 if column > threshold else 0.0
INDICATOR_SPECS = {
    "dpd_flag": ("max_dpd_12m", 0),
}


class RatioAdder(BaseEstimator, TransformerMixin):
    """Add engineered ratio + indicator features, guarding divide-by-zero."""

    def __init__(self, ratios: dict | None = None, indicators: dict | None = None):
        self.ratios = ratios or RATIO_SPECS
        self.indicators = indicators or INDICATOR_SPECS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        for name, (num, den, off) in self.ratios.items():
            numerator = df[num].astype(float)
            denominator = df[den].astype(float) + off
            df[name] = (numerator / denominator).replace([np.inf, -np.inf], np.nan)
        for name, (col, thresh) in self.indicators.items():
            df[name] = (df[col].astype(float) > thresh).astype(float)
        return df


def _columns():
    cfg = get_config().features
    numeric = list(cfg.numeric_features)
    categorical = list(cfg.categorical_features)
    engineered = [
        n for n in cfg.engineered if n in RATIO_SPECS or n in INDICATOR_SPECS
    ]
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
