"""SHAP-based explainability.

The explainers operate on the *transformed* feature matrix (post-pipeline), so
they explain the actual model the system serves. The right explainer is chosen
from the estimator family:
  * tree ensembles (GradientBoosting / RandomForest / HistGradientBoosting)
      -> ``shap.TreeExplainer``
  * linear models (LogisticRegression)
      -> ``shap.LinearExplainer``
  * anything else -> ``shap.Explainer`` (model-agnostic kernel fallback)

All artefacts (global summary PNG, per-row reason codes) are optional outputs —
the serving layer uses :class:`Explainer.explain` for per-request explanations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression

from ..features.engineering import _output_feature_names
from ..utils.logging import get_logger

logger = get_logger(__name__)

_TREE_MODELS = (
    GradientBoostingClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
try:  # xgboost is optional at import time (it's a declared dep but kept defensive)
    from xgboost import XGBClassifier

    _TREE_MODELS = (*_TREE_MODELS, XGBClassifier)
except ImportError:  # pragma: no cover
    pass


@dataclass
class Explanation:
    feature_names: list[str]
    values: np.ndarray              # (n_rows, n_features) SHAP values
    base_value: float
    predictions: np.ndarray         # model probability of default per row


class Explainer:
    """Wrap a fitted pipeline to produce SHAP explanations of its inner model."""

    def __init__(self, pipeline, background: pd.DataFrame, max_background: int = 200):
        self.pipeline = pipeline
        self.preprocessor = pipeline.named_steps["features"]
        self.model = pipeline.named_steps["model"]

        bg = background.iloc[:max_background] if len(background) > max_background else background
        self._background_transformed = self.preprocessor.transform(bg)
        self.feature_names = _output_feature_names(self.preprocessor)
        self._explainer = self._build_explainer()

    def _build_explainer(self):
        if isinstance(self.model, _TREE_MODELS):
            logger.info("Using shap.TreeExplainer (model=%s)", type(self.model).__name__)
            return shap.TreeExplainer(self.model)
        if isinstance(self.model, LogisticRegression):
            logger.info("Using shap.LinearExplainer (model=LogisticRegression)")
            return shap.LinearExplainer(self.model, self._background_transformed)
        logger.info("Using shap.Explainer (generic fallback)")
        return shap.Explainer(self.model, self._background_transformed)

    def explain(self, X: pd.DataFrame) -> Explanation:
        Xt = self.preprocessor.transform(X)
        values = self._explainer.shap_values(Xt)
        # Some tree explainers return a list for binary classifiers — take class-1.
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:  # (n, f, 2) -> class-1 slice
            values = values[:, :, -1]

        base = float(np.ravel(self._explainer.expected_value)[-1])
        preds = np.asarray(self.pipeline.predict_proba(X))[:, 1]
        return Explanation(
            feature_names=list(self.feature_names),
            values=values,
            base_value=base,
            predictions=preds,
        )


def top_reasons(explanation: Explanation, row_index: int, k: int = 5) -> list[dict]:
    """Human-readable top-k contributors for a single prediction."""
    contrib = explanation.values[row_index]
    order = np.argsort(-np.abs(contrib))[:k]
    return [
        {
            "feature": explanation.feature_names[i],
            "contribution": float(contrib[i]),
            "direction": "increases_risk" if contrib[i] > 0 else "decreases_risk",
        }
        for i in order
    ]


def save_global_summary(explanation: Explanation, path: str) -> str:
    """Persist a global SHAP summary plot (best-effort, never fatal)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shap.summary_plot(
        explanation.values,
        features=None,
        feature_names=explanation.feature_names,
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    return path
