"""Deterministic synthetic employee expense claims for anomaly review."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "employee_expense_claims.csv"

FEATURES = [
    "expense_category",
    "department",
    "vendor",
    "payment_method",
    "amount_mwk",
    "receipt_present",
    "duplicate_receipt_flag",
    "submitted_hour",
    "days_after_expense",
    "claims_last_7_days",
    "employee_tenure_months",
    "is_weekend",
]
CATEGORICAL_FEATURES = ["expense_category", "department", "vendor", "payment_method"]
NUMERIC_FEATURES = [feature for feature in FEATURES if feature not in CATEGORICAL_FEATURES]


def generate_dataset(rows: int = 4500, seed: int = 27) -> pd.DataFrame:
    """Create realistic-shaped claims with a small set of unlabelled irregularities."""
    rng = np.random.default_rng(seed)
    categories = np.array(["meals", "travel", "lodging", "supplies", "mileage", "software"])
    departments = np.array(["Operations", "Sales", "Finance", "People", "Technology"])
    vendors = np.array(["Local restaurant", "Regional hotel", "Fuel station", "Office supplier", "Ride service", "Software vendor", "Conference venue", "Airline"])
    payment_methods = np.array(["employee_card", "cash", "mobile_money", "company_card"])
    category = rng.choice(categories, rows, p=[0.24, 0.19, 0.11, 0.18, 0.13, 0.15])
    department = rng.choice(departments, rows, p=[0.28, 0.23, 0.13, 0.12, 0.24])
    vendor = rng.choice(vendors, rows)
    payment_method = rng.choice(payment_methods, rows, p=[0.34, 0.14, 0.20, 0.32])
    typical_amount = {"meals": 11_000, "travel": 56_000, "lodging": 92_000, "supplies": 28_000, "mileage": 19_000, "software": 78_000}
    amount = np.array([rng.lognormal(np.log(typical_amount[name]), 0.52) for name in category]).round(2)
    submitted_hour = np.clip(rng.normal(14, 3.3, rows).round().astype(int), 6, 23)
    days_after_expense = np.clip(rng.gamma(2.2, 3.0, rows).round().astype(int), 0, 45)
    claims_last_7_days = rng.poisson(1.2, rows)
    employee_tenure_months = rng.integers(1, 121, rows)
    receipt_present = rng.binomial(1, 0.94, rows)
    duplicate_receipt_flag = np.zeros(rows, dtype=int)
    is_weekend = rng.binomial(1, 0.10, rows)
    irregularity_count = max(1, int(rows * 0.045))
    irregular = rng.choice(rows, irregularity_count, replace=False)

    for position, row in enumerate(irregular):
        pattern = position % 5
        if pattern == 0:
            amount[row] *= rng.uniform(4.5, 9.0)
        elif pattern == 1:
            duplicate_receipt_flag[row] = 1
        elif pattern == 2:
            receipt_present[row] = 0
            amount[row] *= rng.uniform(2.2, 4.0)
        elif pattern == 3:
            submitted_hour[row] = int(rng.choice([0, 1, 2, 3, 4, 5, 22, 23]))
            claims_last_7_days[row] += int(rng.integers(5, 10))
        else:
            days_after_expense[row] = int(rng.integers(30, 46))
            amount[row] = round(amount[row] / 5000) * 5000

    submitted_dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, rows), unit="D")
    frame = pd.DataFrame(
        {
            "claim_id": [f"CL{index:07d}" for index in range(1, rows + 1)],
            "employee_id": [f"EMP{value:04d}" for value in rng.integers(1, 501, rows)],
            "submitted_date": submitted_dates.strftime("%Y-%m-%d"),
            "expense_category": category,
            "department": department,
            "vendor": vendor,
            "payment_method": payment_method,
            "amount_mwk": amount,
            "receipt_present": receipt_present,
            "duplicate_receipt_flag": duplicate_receipt_flag,
            "submitted_hour": submitted_hour,
            "days_after_expense": days_after_expense,
            "claims_last_7_days": claims_last_7_days,
            "employee_tenure_months": employee_tenure_months,
            "is_weekend": is_weekend,
        }
    )
    return frame


def load_or_generate(path: Path = DATA_PATH, force: bool = False) -> pd.DataFrame:
    if force or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        generate_dataset().to_csv(path, index=False)
    return pd.read_csv(path)