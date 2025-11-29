# email_generator.py
import pandas as pd
from typing import Tuple

from models import CustomerCategory
from templates_db import get_template


def generate_email_for_row(
    row: pd.Series, category: CustomerCategory
) -> Tuple[str, str]:
    """
    Simulates the LLM: pick the right template for the category,
    fill in with customer metadata, and return subject + body.
    """

    template = get_template(category)

    customer_id = row.get("customerID", "Unknown")
    tenure = int(row.get("Tenure", 0))
    monthly = float(row.get("MonthlyCharges", 0) or 0)

    total_raw = row.get("TotalCharges", 0)
    try:
        total = float(total_raw)
    except Exception:
        total = 0.0

    subject = template["subject"].format(
        customer_id=customer_id,
        tenure=tenure,
        monthly_charges=monthly,
        total_charges=total,
    )

    body = template["body"].format(
        customer_id=customer_id,
        tenure=tenure,
        monthly_charges=monthly,
        total_charges=total,
    )

    return subject, body
