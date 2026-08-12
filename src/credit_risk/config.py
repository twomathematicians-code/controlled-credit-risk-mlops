"""Central configuration loader.

The whole system is driven by ``config.yaml`` at the project root. This module
exposes a single cached :func:`get_config` entry point plus a couple of helpers
so every other module reads from one source of truth. Environment variables
override MLflow + serving settings to ease container/Azure deployment.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class Config(dict):
    """A dict that also supports attribute-style access for nested keys."""

    def __getattr__(self, name: str):  # noqa: D401
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value


def _resolve_path(p: str) -> Path:
    """Resolve a config path relative to the project root (unless absolute)."""
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _apply_env_overrides(cfg: dict) -> dict:
    """Let environment variables override MLflow + serving settings."""
    mlflow = cfg.setdefault("mlflow", {})
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow["tracking_uri"] = os.environ["MLFLOW_TRACKING_URI"]
    if os.getenv("MLFLOW_REGISTRY_URI"):
        mlflow["registry_uri"] = os.environ["MLFLOW_REGISTRY_URI"]
    if os.getenv("MLFLOW_EXPERIMENT_NAME"):
        mlflow["experiment_name"] = os.environ["MLFLOW_EXPERIMENT_NAME"]
    if os.getenv("MLFLOW_REGISTRY_MODEL_NAME"):
        mlflow["registry_model_name"] = os.environ["MLFLOW_REGISTRY_MODEL_NAME"]

    serving = cfg.setdefault("serving", {})
    if os.getenv("SERVING_HOST"):
        serving["host"] = os.environ["SERVING_HOST"]
    if os.getenv("SERVING_PORT"):
        serving["port"] = int(os.environ["SERVING_PORT"])
    if os.getenv("SERVING_MODEL_STAGE"):
        serving["model_stage"] = os.environ["SERVING_MODEL_STAGE"]
    return cfg


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Load and cache the parsed config (with env overrides applied)."""
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    _apply_env_overrides(cfg)
    return Config(cfg)


def path(key_path: str) -> Path:
    """Fetch a path-valued config entry, resolved against the project root.

    ``key_path`` uses dotted notation, e.g. ``"data.raw_path"``.
    """
    node: dict | Config = get_config()
    for part in key_path.split("."):
        node = node[part]
    return _resolve_path(node)


def ensure_dirs() -> None:
    """Create the data + artifact directories if they don't yet exist."""
    for key in (
        "data.raw_path",
        "data.processed_path",
        "data.train_path",
        "data.test_path",
        "data.drift_reference_path",
        "monitoring.audit.log_path",
    ):
        try:
            p = path(key)
        except (KeyError, TypeError):
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
