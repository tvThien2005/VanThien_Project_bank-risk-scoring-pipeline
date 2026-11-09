"""Shared API and data-quality validity constraints for raw customer fields."""

from __future__ import annotations

from math import inf

COLUMN_RANGES: dict[str, tuple[float, float]] = {
    "age": (18, 100),
    "income": (0, inf),
    "credit_score": (300, 850),
    "employment_years": (0, inf),
    "account_age_months": (0, inf),
    "monthly_spending": (0, inf),
    "credit_utilization": (0, 1),
    "num_credit_cards": (0, inf),
    "num_loans": (0, inf),
    "late_payment_months": (0, 24),
    "debt_to_income": (0, inf),
    "savings_rate": (0, 1),
    "investment_value": (0, inf),
}

CATEGORICAL_VALUES: dict[str, list[str]] = {
    "employment_status": ["Unemployed", "Part-time", "Self-employed", "Full-time", "Retired"],
    "education_level": ["High School", "Some College", "Bachelor", "Master", "PhD"],
    "marital_status": ["Single", "Married", "Divorced", "Widowed"],
    "home_ownership": ["Rent", "Mortgage", "Own", "Other"],
    "region": ["Northeast", "Southeast", "Midwest", "Southwest", "West"],
}

CUSTOMER_ID_PATTERN = r"^FN\d{7}$"
