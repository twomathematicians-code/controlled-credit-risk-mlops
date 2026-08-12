"""Serving tests — exercise the API with a fitted pipeline injected directly
(keeps these tests fast and MLflow-free)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from credit_risk.serving import app as app_module


def _valid_applicant(small_df):
    row = small_df.iloc[0]
    return {
        "age": int(row["age"]),
        "annual_income": float(row["annual_income"]),
        "months_employed": int(row["months_employed"]),
        "num_open_accounts": int(row["num_open_accounts"]),
        "num_credit_inquiries_12m": int(row["num_credit_inquiries_12m"]),
        "total_debt": float(row["total_debt"]),
        "credit_limit": float(row["credit_limit"]),
        "missed_payments_12m": int(row["missed_payments_12m"]),
        "credit_card_balance": float(row["credit_card_balance"]),
        "employment_status": str(row["employment_status"]),
        "home_ownership": str(row["home_ownership"]),
        "loan_purpose": str(row["loan_purpose"]),
        "region": str(row["region"]),
    }


@pytest.fixture
def client(fitted_pipeline, tmp_path, monkeypatch):
    # Inject the fitted pipeline so endpoints don't need the MLflow registry.
    monkeypatch.setattr(
        app_module,
        "_load_model",
        lambda force=False: (fitted_pipeline, 1, "credit_risk_pd_model"),
    )
    # Redirect the audit log to a temp file.
    def fake_path(key):
        return tmp_path / "audit.jsonl"
    monkeypatch.setattr(app_module.audit, "path", fake_path)
    # Skip SHAP in tests (explanation is best-effort and wrapped in try/except).
    monkeypatch.setattr(app_module, "_get_explainer", lambda pipeline: (_ for _ in ()).throw(RuntimeError("skip")))
    return TestClient(app_module.app)


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_version"] == 1


def test_predict_returns_score_and_decision(client, small_df):
    resp = client.post("/predict", json=_valid_applicant(small_df))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert 0.0 <= body["pd_score"] <= 1.0
    assert body["decision"] in {"APPROVE", "DECLINE"}
    assert body["model_version"] == 1


def test_predict_rejects_invalid_input(client, small_df):
    bad = _valid_applicant(small_df)
    bad["age"] = 5  # below the ge=18 constraint
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_batch(client, small_df):
    payload = {"applicants": [_valid_applicant(small_df) for _ in range(3)]}
    resp = client.post("/predict/batch", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["predictions"]) == 3


def test_health_degraded_without_model(fitted_pipeline, tmp_path, monkeypatch):
    def boom(force=False):
        raise RuntimeError("no model")

    monkeypatch.setattr(app_module, "_load_model", boom)
    resp = TestClient(app_module.app).get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"
