"""Pydantic request and response contracts for the prediction API."""

from __future__ import annotations

from datetime import date
from math import isinf
from typing import Literal

from pydantic import BaseModel, Field, confloat, conint, model_validator

from src.data.validity_rules import (
    CATEGORICAL_VALUES,
    COLUMN_RANGES,
    CUSTOMER_ID_PATTERN,
)


def _integer_range(column: str):
    """Create an integer Pydantic constraint from shared bounds."""
    lower, upper = COLUMN_RANGES[column]
    return conint(ge=int(lower), le=None if isinf(upper) else int(upper))


def _float_range(column: str):
    """Create a floating-point Pydantic constraint from shared bounds."""
    lower, upper = COLUMN_RANGES[column]
    return confloat(ge=lower, le=None if isinf(upper) else upper)


EmploymentStatus = Literal[*CATEGORICAL_VALUES["employment_status"]]
EducationLevel = Literal[*CATEGORICAL_VALUES["education_level"]]
MaritalStatus = Literal[*CATEGORICAL_VALUES["marital_status"]]
HomeOwnership = Literal[*CATEGORICAL_VALUES["home_ownership"]]
Region = Literal[*CATEGORICAL_VALUES["region"]]


class CustomerFeatures(BaseModel):
    """Raw customer values accepted by both saved model bundles."""

    customer_id: str = Field(pattern=CUSTOMER_ID_PATTERN)
    age: _integer_range("age") | None
    income: _float_range("income") | None
    credit_score: _integer_range("credit_score") | None
    employment_years: _float_range("employment_years") | None
    employment_status: EmploymentStatus | None
    education_level: EducationLevel | None
    marital_status: MaritalStatus | None
    home_ownership: HomeOwnership | None
    region: Region | None
    account_age_months: _integer_range("account_age_months") | None
    monthly_spending: _float_range("monthly_spending") | None
    credit_utilization: _float_range("credit_utilization") | None
    num_credit_cards: _integer_range("num_credit_cards") | None
    num_loans: _integer_range("num_loans") | None
    late_payment_months: _integer_range("late_payment_months") | None
    debt_to_income: _float_range("debt_to_income") | None
    savings_rate: _float_range("savings_rate") | None
    investment_value: _float_range("investment_value") | None
    account_creation_date: date | None
    last_transaction_date: date | None

    @model_validator(mode="after")
    def transaction_follows_account_creation(self) -> CustomerFeatures:
        """Reject a transaction date earlier than the account creation date."""
        if (
            self.account_creation_date is not None
            and self.last_transaction_date is not None
            and self.last_transaction_date < self.account_creation_date
        ):
            raise ValueError("last_transaction_date must be on or after account_creation_date.")
        return self


class CreditPrediction(BaseModel):
    """Credit-risk class and class probabilities for one customer."""

    customer_id: str
    credit_risk_prediction: int
    probabilities: dict[str, float]


class FraudPrediction(BaseModel):
    """Fraud score and validation-selected review decision for one customer."""

    customer_id: str
    fraud_risk_score_prediction: float
    review_flag: bool
    review_threshold: float


class BothPrediction(BaseModel):
    """Joint response for the credit and fraud tasks."""

    credit: CreditPrediction
    fraud: FraudPrediction


class HealthResponse(BaseModel):
    """Readiness state reported without requiring prediction access."""

    status: Literal["ok", "model_not_loaded"]
    detail: str | None = None
