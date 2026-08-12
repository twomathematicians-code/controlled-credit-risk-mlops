"""FastAPI scoring service.

Serves the Production-stage model from the MLflow registry. Every prediction is
explained (SHAP reason codes) and recorded in the audit log. Monitoring
endpoints expose drift and performance checks so the same service is the single
source of truth for inference *and* model health.

Run: ``python -m credit_risk.serving.app``
"""
from __future__ import annotations

import sys
import threading

import pandas as pd
from fastapi import FastAPI, HTTPException

from ..config import PROJECT_ROOT, get_config
from ..models import registry
from ..monitoring import audit, drift, performance
from ..utils.logging import get_logger
from . import schemas

logger = get_logger(__name__)

app = FastAPI(
    title="Credit Risk PD Scoring API",
    description="Controlled, production-grade probability-of-default scoring with MLflow registry, "
    "drift/performance monitoring, audit logging and SHAP explainability.",
    version="0.1.0",
)

_lock = threading.Lock()
_model_cache: dict = {"pipeline": None, "version": None, "name": None}
_explainer = {"obj": None}


# ---------------------------------------------------------------------------
# Lazy model + explainer loading
# ---------------------------------------------------------------------------
def _load_model(force: bool = False):
    with _lock:
        if not force and _model_cache["pipeline"] is not None:
            return _model_cache["pipeline"], _model_cache["version"], _model_cache["name"]
        name = get_config().mlflow.registry_model_name
        try:
            pipeline = registry.load_production_model(name)
            version = registry.production_version(name)
        except Exception as exc:
            logger.warning("Could not load Production model '%s': %s", name, exc)
            raise
        _model_cache.update({"pipeline": pipeline, "version": version, "name": name})
        logger.info("Loaded Production model '%s' version %s", name, version)
        _explainer["obj"] = None  # invalidate explainer when model changes
        return pipeline, version, name


def _get_explainer(pipeline):
    if _explainer["obj"] is not None:
        return _explainer["obj"]
    from ..models.explainability import Explainer

    ref_path = PROJECT_ROOT / "data" / "drift_reference" / "reference.parquet"
    background = pd.read_parquet(ref_path)
    background = background.drop(columns=[get_config().data.target_column], errors="ignore")
    explainer = Explainer(pipeline, background=background)
    _explainer["obj"] = explainer
    return explainer


def _decision_threshold() -> float:
    """Prefer the cost-optimal threshold from training; fall back to config."""
    metrics_path = PROJECT_ROOT / "data" / "processed" / "model_metrics.json"
    if metrics_path.exists():
        try:
            import json

            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            return float(data.get("optimal_threshold", get_config().serving.default_threshold))
        except Exception:  # pragma: no cover
            pass
    return float(get_config().serving.default_threshold)


def _to_df(applicants: list[schemas.Applicant]) -> pd.DataFrame:
    return pd.DataFrame([a.model_dump() for a in applicants])


def _decide(pd_score: float, threshold: float) -> str:
    return (
        get_config().serving.decision_labels.decline
        if pd_score >= threshold
        else get_config().serving.decision_labels.approve
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=schemas.HealthResponse, tags=["system"])
def health() -> schemas.HealthResponse:
    try:
        _, version, name = _load_model()
        status = "ok"
    except Exception:
        version, name = None, get_config().mlflow.registry_model_name
        status = "degraded"
    return schemas.HealthResponse(
        status=status,
        model_name=name,
        model_version=version,
        stage=get_config().serving.model_stage,
        threshold=_decision_threshold(),
    )


@app.post("/predict", response_model=schemas.PredictionResponse, tags=["scoring"])
def predict(applicant: schemas.Applicant) -> schemas.PredictionResponse:
    return _predict_batch([applicant])[0]


@app.post("/predict/batch", response_model=schemas.BatchResponse, tags=["scoring"])
def predict_batch(request: schemas.BatchRequest) -> schemas.BatchResponse:
    predictions = _predict_batch(request.applicants)
    drift_status = None
    if len(request.applicants) >= 20:
        try:
            drift_status = drift.evaluate(_to_df(request.applicants))
        except Exception as exc:  # pragma: no cover
            logger.warning("Drift check skipped: %s", exc)
    return schemas.BatchResponse(predictions=predictions, drift_status=drift_status)


def _predict_batch(applicants: list[schemas.Applicant]) -> list[schemas.PredictionResponse]:
    try:
        pipeline, version, name = _load_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"No Production model available: {exc}") from exc

    df = _to_df(applicants)
    threshold = _decision_threshold()
    proba = pipeline.predict_proba(df)[:, 1]

    # Per-prediction SHAP reasons (best-effort).
    reasons_per_row: list[list[dict]] = [[] for _ in range(len(applicants))]
    try:
        from ..models.explainability import top_reasons

        explainer = _get_explainer(pipeline)
        explanation = explainer.explain(df)
        reasons_per_row = [top_reasons(explanation, i, k=4) for i in range(len(df))]
    except Exception as exc:  # pragma: no cover
        logger.warning("Explanation skipped: %s", exc)

    responses: list[schemas.PredictionResponse] = []
    for i, applicant in enumerate(applicants):
        pd_score = float(proba[i])
        decision = _decide(pd_score, threshold)
        audit.record(
            model_name=name,
            model_version=version,
            features=applicant.model_dump(),
            pd_score=pd_score,
            decision=decision,
            threshold=threshold,
            reasons=reasons_per_row[i],
            actor="api",
        )
        responses.append(
            schemas.PredictionResponse(
                application_id=getattr(applicant, "application_id", None),
                pd_score=round(pd_score, 6),
                decision=decision,
                threshold=threshold,
                model_name=name,
                model_version=version,
                reasons=reasons_per_row[i],
            )
        )
    return responses


@app.post("/monitor/drift", tags=["monitoring"])
def monitor_drift(request: schemas.BatchRequest) -> dict:
    """Compare the supplied batch against the frozen reference distribution."""
    current = _to_df(request.applicants)
    return drift.evaluate(current)


@app.post("/monitor/performance", tags=["monitoring"])
def monitor_performance(request: schemas.PerformanceRequest) -> dict:
    """Recompute realised metrics for a labelled batch and emit alerts."""
    if len(request.applicants) != len(request.default_flags):
        raise HTTPException(status_code=422, detail="applicants and default_flags must align")
    try:
        pipeline, version, name = _load_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"No Production model available: {exc}") from exc
    scores = pipeline.predict_proba(_to_df(request.applicants))[:, 1]
    return performance.evaluate(request.default_flags, scores)


@app.get("/metrics", tags=["monitoring"])
def metrics() -> dict:
    """Lightweight operational snapshot: model pin + audit volume."""
    try:
        _, version, name = _load_model()
    except Exception:
        version, name = None, get_config().mlflow.registry_model_name
    return {
        "model_name": name,
        "model_version": version,
        "audit_records": len(audit.read()),
        "threshold": _decision_threshold(),
    }


def main() -> None:  # pragma: no cover
    import uvicorn

    cfg = get_config().serving
    uvicorn.run("credit_risk.serving.app:app", host=cfg.host, port=int(cfg.port), reload=False)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
