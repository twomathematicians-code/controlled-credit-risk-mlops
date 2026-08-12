"""Model training, selection, registration and governance logging.

Pipeline (each step is logged to MLflow for auditability):
  1. Load processed train/test splits.
  2. For every candidate estimator: stratified k-fold CV across multiple metrics.
  3. Bootstrap a 95% CI on the test ROC-AUC (statistical validation signal).
  4. Select the champion by mean CV ROC-AUC; fit it on the full training set.
  5. Compute business KPIs (expected loss, cost-optimal threshold).
  6. Log params, metrics, artifacts (config, threshold curve, SHAP summary).
  7. Register the champion and (optionally) promote it to Production.

Run: ``python -m credit_risk.models.train``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mlflow
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from .. import business
from ..config import PROJECT_ROOT, get_config
from ..features.engineering import build_pipeline
from ..utils.io import read_parquet
from ..utils.logging import get_logger
from . import registry

logger = get_logger(__name__)

METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "model_metrics.json"


def build_estimators() -> dict:
    root = get_config()
    cfg = root.model
    seed = int(root.random_seed)
    estimators = {}
    if "logreg" in cfg.candidates:
        estimators["logreg"] = LogisticRegression(
            C=float(cfg.logreg.C), max_iter=int(cfg.logreg.max_iter), random_state=seed
        )
    if "gbdt" in cfg.candidates:
        estimators["gbdt"] = GradientBoostingClassifier(
            n_estimators=int(cfg.gbdt.n_estimators),
            learning_rate=float(cfg.gbdt.learning_rate),
            max_depth=int(cfg.gbdt.max_depth),
            subsample=float(cfg.gbdt.subsample),
            random_state=int(cfg.gbdt.random_state),
        )
    if "xgboost" in cfg.candidates:
        from xgboost import XGBClassifier

        estimators["xgboost"] = XGBClassifier(
            n_estimators=int(cfg.xgboost.n_estimators),
            learning_rate=float(cfg.xgboost.learning_rate),
            max_depth=int(cfg.xgboost.max_depth),
            subsample=float(cfg.xgboost.subsample),
            colsample_bytree=float(cfg.xgboost.colsample_bytree),
            random_state=int(cfg.xgboost.random_state),
            eval_metric="logloss",
            tree_method="hist",
        )
    return estimators


def _feature_columns() -> list[str]:
    cfg = get_config().data
    return list(cfg.numeric_features) + list(cfg.categorical_features)


def _cv_scores(estimator, X, y, seed: int, folds: int) -> dict:
    scoring = ["roc_auc", "average_precision", "f1", "recall", "precision"]
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    results = cross_validate(
        build_pipeline(estimator), X, y, cv=cv, scoring=scoring, n_jobs=1, error_score="raise"
    )
    return {m: (float(results[f"test_{m}"].mean()), float(results[f"test_{m}"].std())) for m in scoring}


def bootstrap_auc_ci(y_true, y_score, n_boot: int = 300, seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap 95% CI on ROC-AUC. Returns (point_estimate, lower, upper)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    point = roc_auc_score(y_true, y_score)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    lower, upper = np.percentile(aucs, [2.5, 97.5])
    return float(point), float(lower), float(upper)


def run(register: bool = True, promote: bool = True) -> dict:
    cfg = get_config()
    seed = int(cfg.random_seed)
    registry.configure_mlflow()

    train = read_parquet(cfg.data.train_path if (Path(cfg.data.train_path).is_absolute())
                         else str(PROJECT_ROOT / cfg.data.train_path))
    test = read_parquet(cfg.data.test_path if (Path(cfg.data.test_path).is_absolute())
                        else str(PROJECT_ROOT / cfg.data.test_path))
    feat_cols = _feature_columns()
    target = cfg.data.target_column
    X_train, y_train = train[feat_cols], train[target]
    X_test, y_test = test[feat_cols], test[target]

    estimators = build_estimators()
    results: dict = {"candidates": {}}

    champion_name, champion_cv = None, -1.0
    for name, est in estimators.items():
        cv = _cv_scores(est, X_train, y_train, seed, int(cfg.model.cv_folds))
        logger.info("[%s] CV roc_auc=%.4f ± %.4f", name, cv["roc_auc"][0], cv["roc_auc"][1])
        results["candidates"][name] = {"cv": cv}
        if cv["roc_auc"][0] > champion_cv:
            champion_cv, champion_name = cv["roc_auc"][0], name

    logger.info("Champion candidate: %s (CV roc_auc=%.4f)", champion_name, champion_cv)
    champion = build_pipeline(estimators[champion_name])
    champion.fit(X_train, y_train)

    test_proba = champion.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_proba)
    test_ap = average_precision_score(y_test, test_proba)
    auc_point, auc_lo, auc_hi = bootstrap_auc_ci(y_test, test_proba, seed=seed + 7)

    # Business KPIs + cost-optimal threshold
    opt_threshold, opt_metrics = business.optimal_threshold(y_test.to_numpy(), test_proba)
    threshold_curve = business.threshold_curve(y_test.to_numpy(), test_proba)
    curve_path = PROJECT_ROOT / "data" / "processed" / "threshold_curve.csv"
    threshold_curve.to_csv(curve_path, index=False)

    results.update(
        {
            "champion": champion_name,
            "test_roc_auc": test_auc,
            "test_average_precision": test_ap,
            "test_roc_auc_ci": [auc_lo, auc_hi],
            "optimal_threshold": opt_threshold,
            "optimal_threshold_metrics": opt_metrics,
        }
    )

    # ---- MLflow logging ----
    with mlflow.start_run(run_name=f"{champion_name}_{seed}") as run:
        mlflow.log_params({"champion": champion_name, "seed": seed, "cv_folds": cfg.model.cv_folds})
        for metric, (mean, std) in results["candidates"][champion_name]["cv"].items():
            mlflow.log_metric(f"cv_{metric}_mean", mean)
            mlflow.log_metric(f"cv_{metric}_std", std)
        mlflow.log_metric("test_roc_auc", test_auc)
        mlflow.log_metric("test_average_precision", test_ap)
        mlflow.log_metric("test_roc_auc_ci_lower", auc_lo)
        mlflow.log_metric("test_roc_auc_ci_upper", auc_hi)
        mlflow.log_metric("optimal_threshold", opt_threshold)
        mlflow.log_metric("optimal_total_cost", opt_metrics["total_cost"])
        mlflow.log_metric("optimal_approval_rate", opt_metrics["approval_rate"])

        mlflow.sklearn.log_model(
            champion,
            artifact_path="model",
            serialization_format="cloudpickle",
        )
        mlflow.log_artifact(str(PROJECT_ROOT / "config.yaml"))
        mlflow.log_artifact(str(curve_path))
        _log_shap_summary(champion, X_train, X_test, run.info.run_id)

        if register:
            version = registry.register_pipeline(run.info.run_id)
            results["registered_version"] = version
            if promote:
                registry.promote_version_to_production(version)
                results["promoted_to_production"] = True

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote metrics -> %s", METRICS_PATH)
    logger.info(
        "Done | champion=%s test_auc=%.4f [%.4f, %.4f] opt_threshold=%.3f",
        champion_name, test_auc, auc_lo, auc_hi, opt_threshold,
    )
    return results


def _log_shap_summary(pipeline, X_train, X_test, run_id: str) -> None:
    """Best-effort SHAP summary artefact; never fatal to training."""
    try:
        from .explainability import Explainer, save_global_summary

        explainer = Explainer(pipeline, background=X_train)
        explanation = explainer.explain(X_test.iloc[: min(500, len(X_test))])
        summary_path = PROJECT_ROOT / "data" / "processed" / "shap_summary.png"
        save_global_summary(explanation, str(summary_path))
        mlflow.log_artifact(str(summary_path))
        logger.info("Logged SHAP summary artefact")
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("SHAP summary skipped: %s", exc)


def main() -> None:  # pragma: no cover
    run()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
