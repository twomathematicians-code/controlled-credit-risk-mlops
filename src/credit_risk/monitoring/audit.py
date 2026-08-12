"""Append-only structured audit logging.

Every scoring decision is recorded as one JSON line so the system has a
tamper-evident trail of *who/what/when*: model name + version, timestamp, a hash
of the input features, the probability, the decision and the reason codes.
This is the backbone of the "controlled production environment" story.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from ..config import path
from ..utils.io import feature_hash
from ..utils.logging import get_logger

logger = get_logger(__name__)
_lock = Lock()


def _log_path() -> Path:
    return path("monitoring.audit.log_path")


def record(
    *,
    model_name: str,
    model_version: int | str | None,
    features: dict[str, Any],
    pd_score: float,
    decision: str,
    threshold: float,
    reasons: list[dict] | None = None,
    actor: str = "system",
    extra: dict | None = None,
) -> dict:
    """Append one audit record. Returns the record that was written."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "actor": actor,
        "model_name": model_name,
        "model_version": model_version,
        "feature_hash": feature_hash(features),
        "pd_score": round(float(pd_score), 6),
        "decision": decision,
        "threshold": float(threshold),
        "reasons": reasons or [],
    }
    if extra:
        entry["extra"] = extra

    log_path = _log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, default=str)
    with _lock, log_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return entry


def read(limit: int | None = None) -> list[dict]:
    """Read audit records (most recent last)."""
    log_path = _log_path()
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8").splitlines()
    if limit:
        lines = lines[-limit:]
    return [json.loads(line) for line in lines if line.strip()]
