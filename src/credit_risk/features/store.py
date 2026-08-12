"""Lightweight feature store.

Not a distributed store — a small, well-documented abstraction that captures the
*spirit* of a feature store: versioned, named feature groups with a manifest, so
training and serving consume identical, reproducible features.

Layout (under ``features.store_dir``)::

    feature_store/
      manifest.json          # group -> {path, columns, row_count, created_at, sha}
      train_features.parquet
      test_features.parquet
      reference_features.parquet
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import get_config, path
from ..utils.io import read_parquet, write_parquet
from ..utils.logging import get_logger

logger = get_logger(__name__)
MANIFEST_NAME = "manifest.json"


def _store_dir() -> Path:
    d = path("features.store_dir")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest_path() -> Path:
    return _store_dir() / MANIFEST_NAME


def _read_manifest() -> dict:
    p = _manifest_path()
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _write_manifest(manifest: dict) -> None:
    _manifest_path().write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def _sha(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]


def put(name: str, df: pd.DataFrame) -> Path:
    """Persist a feature group snapshot and register it in the manifest."""
    out = _store_dir() / f"{name}.parquet"
    write_parquet(df, out)
    manifest = _read_manifest()
    manifest[name] = {
        "path": str(out.relative_to(path("features.store_dir").parent.parent)),
        "columns": list(df.columns),
        "row_count": int(len(df)),
        "sha": _sha(df),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_manifest(manifest)
    logger.info("Feature store: wrote '%s' (%d rows, sha=%s)", name, len(df), manifest[name]["sha"])
    return out


def get(name: str) -> pd.DataFrame:
    """Load a feature group snapshot by name."""
    manifest = _read_manifest()
    if name not in manifest:
        raise KeyError(f"Feature group '{name}' not found in store. Available: {list(manifest)}")
    return read_parquet(_store_dir() / f"{name}.parquet")


def list_groups() -> dict:
    """Return the manifest (group -> metadata)."""
    return _read_manifest()


def snapshot() -> dict:
    """Materialise train/test/reference feature groups from processed splits."""

    cfg = get_config().data
    train = read_parquet(path("data.train_path"))
    test = read_parquet(path("data.test_path"))
    reference = read_parquet(path("data.drift_reference_path"))
    feature_cols = list(cfg.numeric_features) + list(cfg.categorical_features)
    groups = {
        "train_features": train[feature_cols + [cfg.target_column]],
        "test_features": test[feature_cols + [cfg.target_column]],
        "reference_features": reference,
    }
    return {name: put(name, df) for name, df in groups.items()}
