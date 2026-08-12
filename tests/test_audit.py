"""Audit logging tests."""
from __future__ import annotations

import pytest

from credit_risk.monitoring import audit


@pytest.fixture(autouse=True)
def isolated_log(tmp_path, monkeypatch):
    """Redirect the audit log to a temp file so tests don't pollute real state."""
    log_file = tmp_path / "audit.jsonl"

    def fake_path(key):
        return log_file

    monkeypatch.setattr(audit, "path", fake_path)
    return log_file


def test_record_and_read_roundtrip(isolated_log):
    entry = audit.record(
        model_name="m",
        model_version=3,
        features={"age": 40, "annual_income": 50000},
        pd_score=0.23,
        decision="APPROVE",
        threshold=0.5,
        reasons=[{"feature": "x", "contribution": 0.1, "direction": "increases_risk"}],
    )
    assert entry["feature_hash"]
    records = audit.read()
    assert len(records) == 1
    assert records[0]["decision"] == "APPROVE"
    assert records[0]["model_version"] == 3


def test_feature_hash_is_deterministic(isolated_log):
    audit.record(model_name="m", model_version=1, features={"a": 1}, pd_score=0.1,
                 decision="APPROVE", threshold=0.5)
    audit.record(model_name="m", model_version=1, features={"a": 1}, pd_score=0.1,
                 decision="APPROVE", threshold=0.5)
    recs = audit.read()
    assert recs[0]["feature_hash"] == recs[1]["feature_hash"]
