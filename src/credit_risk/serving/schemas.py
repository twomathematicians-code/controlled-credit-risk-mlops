"""Pydantic request/response schemas — the strict, documented API contract."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Applicant(BaseModel):
    """A single credit applicant (raw scoring features)."""

    age: int = Field(..., ge=18, le=100, description="Applicant age in years")
    annual_income: float = Field(..., gt=0, description="Reported annual income")
    months_employed: int = Field(..., ge=0, description="Months in current employment")
    num_open_accounts: int = Field(..., ge=0, description="Number of open credit accounts")
    num_credit_inquiries_12m: int = Field(..., ge=0, description="Hard inquiries in the last 12 months")
    total_debt: float = Field(..., ge=0, description="Total outstanding debt")
    credit_limit: float = Field(..., ge=0, description="Total credit card limit")
    missed_payments_12m: int = Field(..., ge=0, description="Missed payments in the last 12 months")
    credit_card_balance: float = Field(..., ge=0, description="Current credit card balance")
    employment_status: str
    home_ownership: str
    loan_purpose: str
    region: str


class PredictionResponse(BaseModel):
    application_id: int | None = None
    pd_score: float
    decision: str
    threshold: float
    model_name: str
    model_version: int | None = None
    reasons: list[dict] = []


class BatchRequest(BaseModel):
    applicants: list[Applicant]


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    drift_status: dict | None = None


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: int | None = None
    stage: str
    threshold: float


class PerformanceRequest(BaseModel):
    """Used by /monitor/performance — applicants plus realised default flags."""

    applicants: list[Applicant]
    default_flags: list[int] = Field(..., description="1 if the applicant defaulted, else 0")
