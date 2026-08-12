"""Pydantic request/response schemas — the strict, documented API contract."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Applicant(BaseModel):
    """A single credit applicant (raw scoring features, Home Credit schema)."""

    age: int = Field(..., ge=18, le=100, description="Applicant age in years")
    income: float = Field(..., gt=0, description="Occupation income")
    credit_amount: float = Field(..., ge=0, description="Credit amount applied for")
    total_debt: float = Field(..., ge=0, description="Total outstanding debt")
    current_debt: float = Field(..., ge=0, description="Current debt")
    num_active_credits: int = Field(..., ge=0, description="Number of active credits")
    num_credit_inquiries: int = Field(..., ge=0, description="Number of credit queries")
    recent_applications: int = Field(..., ge=0, description="Recent application count")
    max_dpd_12m: int = Field(..., ge=0, description="Max days past due in last 12 months")
    num_installments: int = Field(..., ge=0, description="Number of installments")
    sex: str
    education: str
    income_type: str
    family_status: str
    employment_duration: str


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
