# classifier.py
import pandas as pd
from models import CustomerCategory


def classify_customer(row: pd.Series) -> CustomerCategory:
    """
    Simple rule-based classifier using columns from TelecomCustomerChurn.csv

    Assumes the CSV has at least:
    - Tenure
    - MonthlyCharges
    - TotalCharges
    - Churn (Yes/No)

    Rules (you can tweak):
    - critical:
        churn == "No" AND (tenure >= 24 OR total_charges >= 2000 OR monthly_charges >= 80)
    - occasional_defaulter:
        churn == "No" AND tenure >= 6
    - habitual_defaulter:
        everything else
    """
    tenure = int(row.get("Tenure", 0))
    monthly = float(row.get("MonthlyCharges", 0) or 0)

    total_raw = row.get("TotalCharges", 0)
    try:
        total = float(total_raw)
    except Exception:
        total = 0.0

    churn = str(row.get("Churn", "No"))

    if churn == "No" and (tenure >= 24 or total >= 2000 or monthly >= 80):
        return "critical"
    elif churn == "No" and tenure >= 6:
        return "occasional_defaulter"
    else:
        return "habitual_defaulter"
