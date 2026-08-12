"""MLflow model registry layer.

Wraps MLflow so the rest of the codebase never touches MLflow directly. Handles:
  * tracking/registry URI + experiment configuration (idempotent)
  * creating registered models + versions
  * promoting a version to "Production" via the *alias* API (MLflow >= 2.9 / 3.x),
    with a transparent fallback to the legacy stage API on older installs
  * loading the Production model for serving (alias-first, stage-fallback)

Only the Production-tagged version is ever served — promotion is an explicit,
logged governance action, not automatic.
"""
from __future__ import annotations

import contextlib

import mlflow
from mlflow.tracking import MlflowClient

from ..config import PROJECT_ROOT, get_config
from ..utils.logging import get_logger

logger = get_logger(__name__)
PRODUCTION_ALIAS = "Production"


def configure_mlflow() -> MlflowClient:
    """Configure tracking + registry URIs and ensure the experiment exists."""
    cfg = get_config().mlflow
    mlflow.set_tracking_uri(cfg.tracking_uri)
    with contextlib.suppress(Exception):  # pragma: no cover - older mlflow
        mlflow.set_registry_uri(cfg.registry_uri)

    client = MlflowClient()
    exp = client.get_experiment_by_name(cfg.experiment_name)
    if exp is None:
        # Use a proper file:// URI (forward slashes) so the artifact store resolves
        # correctly on Windows as well as POSIX.
        artifact_location = (PROJECT_ROOT / cfg.artifact_location).resolve().as_uri()
        client.create_experiment(cfg.experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(cfg.experiment_name)
    return client


def get_or_create_registered_model(client: MlflowClient, name: str | None = None) -> str:
    name = name or get_config().mlflow.registry_model_name
    try:
        client.get_registered_model(name)
    except Exception:
        client.create_registered_model(name)
        logger.info("Created registered model '%s'", name)
    return name


def register_pipeline(run_id: str, name: str | None = None) -> int:
    """Register the model artifact from a run as a new version. Returns version."""
    configure_mlflow()
    name = name or get_config().mlflow.registry_model_name
    client = MlflowClient()
    get_or_create_registered_model(client, name)
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=name)
    version = int(result.version)
    logger.info("Registered '%s' version %d (run %s)", name, version, run_id)
    return version


def _set_production(client: MlflowClient, name: str, version: int) -> None:
    """Promote a version to Production — alias API first, stage fallback."""
    try:
        client.set_registered_model_alias(name, PRODUCTION_ALIAS, version)
        logger.info("Promoted '%s' v%d via alias '%s'", name, version, PRODUCTION_ALIAS)
        return
    except Exception as exc:  # pragma: no cover - fallback for older mlflow
        logger.warning("Alias promotion failed (%s); falling back to stage API", exc)
    try:
        client.transition_model_version_stage(
            name=name, version=version, stage=PRODUCTION_ALIAS, archive_existing_versions=True
        )
        logger.info("Promoted '%s' v%d via stage '%s'", name, version, PRODUCTION_ALIAS)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Could not promote '{name}' v{version} to Production: {exc}") from exc


def promote_version_to_production(version: int, name: str | None = None) -> int:
    client = configure_mlflow()
    name = name or get_config().mlflow.registry_model_name
    get_or_create_registered_model(client, name)
    _set_production(client, name, version)
    return version


def latest_version(name: str | None = None) -> int:
    client = configure_mlflow()
    name = name or get_config().mlflow.registry_model_name
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise RuntimeError(f"No registered versions found for '{name}'.")
    return max(int(v.version) for v in versions)


def promote_latest_to_production(name: str | None = None) -> int:
    version = latest_version(name)
    return promote_version_to_production(version, name)


def load_production_model(name: str | None = None):
    """Load the Production model — alias URI first, stage fallback."""
    configure_mlflow()
    name = name or get_config().mlflow.registry_model_name
    alias_uri = f"models:/{name}@{PRODUCTION_ALIAS}"
    try:
        return mlflow.sklearn.load_model(alias_uri)
    except Exception as exc:  # pragma: no cover - fallback for older mlflow
        logger.warning("Alias load failed (%s); trying stage URI", exc)
        return mlflow.sklearn.load_model(f"models:/{name}/{PRODUCTION_ALIAS}")


def production_version(name: str | None = None) -> int | None:
    """Resolve the current Production version (via alias), or None if not set."""
    client = configure_mlflow()
    name = name or get_config().mlflow.registry_model_name
    try:
        mv = client.get_model_version_by_alias(name, PRODUCTION_ALIAS)
        return int(mv.version)
    except Exception:
        return None
