import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Constants
N_CUSTOMERS = 10000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def generate_telecom_data(n=N_CUSTOMERS):
    """
    Generates a synthetic telecom customer dataset with n rows.
    - ONLY POSTPAID customers
    - Includes payment behaviour + usage + profile features
    - Segment labels: critical, habitual_defaulter, occasional_defaulter
    """
    print(f"Generating data for {n} customers...")

    # --- 1. BASIC TELECOM INFO (ONLY POSTPAID) ---
    customer_ids = [f"C{str(i).zfill(4)}" for i in range(1, n + 1)]
    plan_types = np.array(["postpaid"] * n)   # ✅ only postpaid

    is_family_plan = np.random.choice([0, 1], size=n, p=[0.8, 0.2])

    # Avg monthly bill (skewed, clipped)
    avg_monthly_bill = np.random.lognormal(mean=6.5, sigma=0.6, size=n)
    avg_monthly_bill = np.clip(avg_monthly_bill, 200, 3000).round(2)

    # Tenure
    tenure_months = np.random.randint(1, 121, size=n)

    months_active_12m = np.minimum(12, tenure_months)
    noise = np.random.normal(0, 50, size=n)
    total_revenue_12m = (avg_monthly_bill * months_active_12m + noise)
    total_revenue_12m = np.maximum(0, total_revenue_12m).round(2)

    # --- 2. PAYMENT BEHAVIOUR (FEATURES, NOT LABEL) ---
    # Base rates; will later influence risk score
    late_payments_12m = np.random.poisson(lam=1.5, size=n)
    late_payments_12m = np.minimum(late_payments_12m, 12)

    missed_payments_12m = np.random.poisson(lam=0.5, size=n)
    missed_payments_12m = np.minimum(missed_payments_12m, 8)

    # Max days late
    max_days_late = np.zeros(n, dtype=int)
    mask_late = late_payments_12m > 0
    max_days_late[mask_late] = np.random.randint(1, 30, size=mask_late.sum())
    mask_missed = missed_payments_12m > 0
    max_days_late[mask_missed] = np.random.randint(30, 121, size=mask_missed.sum())

    # Total bills in last 12m
    num_bills = months_active_12m.copy()

    # Fix counts not to exceed bills
    late_payments_12m = np.minimum(late_payments_12m, num_bills)
    missed_payments_12m = np.minimum(missed_payments_12m, num_bills)

    remaining = num_bills - missed_payments_12m
    late_payments_12m = np.minimum(late_payments_12m, remaining)
    num_payments_on_time_12m = num_bills - late_payments_12m - missed_payments_12m
    num_payments_late_12m = late_payments_12m

    safe_total = np.where(num_bills > 0, num_bills, 1)
    late_payment_ratio = (num_payments_late_12m / safe_total).round(2)
    missed_payment_ratio = (missed_payments_12m / safe_total).round(2)

    # Avg delay
    avg_delay = (
        num_payments_late_12m * np.random.uniform(5, 25, size=n) +
        missed_payments_12m * np.random.uniform(30, 60, size=n)
    )
    avg_payment_delay_days = (avg_delay / safe_total).round(1)

    # Last payment
    last_payment_status = np.random.choice(
        ["on_time", "late", "missed"], size=n, p=[0.7, 0.2, 0.1]
    )
    last_payment_delay_days = np.zeros(n, dtype=int)
    mask_last_late = last_payment_status == "late"
    mask_last_missed = last_payment_status == "missed"
    last_payment_delay_days[mask_last_late] = np.random.randint(1, 30, size=mask_last_late.sum())
    last_payment_delay_days[mask_last_missed] = np.random.randint(30, 90, size=mask_last_missed.sum())

    # Overdue flags
    days_overdue_current_bill = np.zeros(n, dtype=int)
    is_currently_overdue = np.zeros(n, dtype=int)
    # some portion of those with late/missed are overdue
    overdue_mask = (late_payments_12m + missed_payments_12m) > 0
    days_overdue_current_bill[overdue_mask] = np.random.randint(1, 61, size=overdue_mask.sum())
    is_currently_overdue[days_overdue_current_bill > 0] = 1

    # --- 3. BILLING & PAYMENT METHOD ---
    payment_method = np.random.choice(
        ["bank_transfer", "credit_card", "UPI", "cash"],
        size=n
    )

    auto_pay_enabled = np.random.choice([0, 1], size=n, p=[0.6, 0.4])
    mask_good = missed_payments_12m == 0
    auto_pay_enabled[mask_good] = np.random.choice(
        [0, 1], size=mask_good.sum(), p=[0.3, 0.7]
    )

    billing_cycle_day = np.random.randint(1, 31, size=n)
    historical_billing_disputes = np.random.poisson(0.5, size=n)

    # --- 4. CUSTOMER PROFILE & DEMOGRAPHICS ---
    customer_age = np.random.randint(18, 76, size=n)
    customer_region = np.random.choice(
        ["north", "south", "east", "west", "metro", "rural"],
        size=n
    )
    occupation_type = np.random.choice(
        ["student", "salaried", "self_employed", "unemployed", "retired"],
        size=n,
        p=[0.1, 0.4, 0.3, 0.05, 0.15]
    )
    income_bracket = np.random.choice(
        ["low", "lower_middle", "upper_middle", "high"],
        size=n,
        p=[0.2, 0.4, 0.3, 0.1]
    )
    account_type = np.random.choice(
        ["individual", "SME", "corporate"],
        size=n,
        p=[0.85, 0.10, 0.05]
    )
    num_dependents = np.random.randint(0, 6, size=n)

    # --- 5. TELECOM USAGE ---
    base_usage_gb = avg_monthly_bill / 20
    avg_monthly_data_usage_gb = np.random.normal(base_usage_gb, 10, size=n).clip(1, 200).round(1)

    avg_monthly_voice_minutes = np.random.randint(50, 3001, size=n)
    avg_monthly_sms = np.random.randint(0, 1001, size=n)
    international_usage_minutes = np.random.exponential(10, size=n).clip(0, 500).round(0)
    roaming_usage_minutes = np.random.exponential(20, size=n).clip(0, 1000).round(0)
    value_added_services_subscribed = np.random.choice([0, 1], size=n, p=[0.7, 0.3])

    # --- 6. SUBSCRIPTION & PLAN METADATA ---
    plan_category = np.random.choice(
        ["basic", "standard", "premium"],
        size=n,
        p=[0.4, 0.4, 0.2]
    )
    plan_category[avg_monthly_bill > 1500] = "premium"

    contract_length_months = np.random.choice([1, 3, 6, 12, 24], size=n)
    discounts_or_offers_applied = np.random.choice([0, 1], size=n, p=[0.6, 0.4])
    plan_change_count_last_12m = np.random.poisson(0.3, size=n).clip(0, 5)
    is_5g_plan = np.random.choice([0, 1], size=n, p=[0.5, 0.5])

    num_lines_in_account = np.random.choice(
        [1, 2, 3, 4, 5], size=n, p=[0.7, 0.15, 0.08, 0.05, 0.02]
    )
    num_lines_in_account[is_family_plan == 1] = np.random.choice(
        [2, 3, 4, 5], size=(is_family_plan == 1).sum()
    )

    # --- 7. SUPPORT HISTORY ---
    num_support_calls_12m = np.random.poisson(2, size=n)
    complaint_count_12m = np.random.poisson(0.5, size=n)
    issue_type_majority = np.random.choice(
        ["billing", "network", "other"], size=n
    )
    refund_requests_12m = np.random.poisson(0.1, size=n)
    dispute_count_12m = np.random.poisson(0.1, size=n)

    # --- 8. ENGAGEMENT & LOYALTY ---
    loyalty_score = np.random.randint(1, 11, size=n)
    loyalty_score[tenure_months > 24] = np.random.randint(
        5, 11, size=(tenure_months > 24).sum()
    )

    app_login_frequency_monthly = np.random.randint(0, 61, size=n)
    reward_points_balance = np.random.randint(0, 10001, size=n) * loyalty_score
    renewal_rate = np.random.uniform(0.1, 1.0, size=n).round(2)

    # --- 9. FINANCIAL INDICATORS ---
    credit_score_range = np.random.choice(
        ["good", "fair", "poor"],
        size=n,
        p=[0.4, 0.4, 0.2]
    )

    wallet_balance = np.random.uniform(0, 5000, size=n).round(2)
    avg_topup_amount = np.random.uniform(50, 2000, size=n).round(2)
    topup_frequency = np.random.randint(0, 31, size=n)

    max_payment_delay_days_12m = max_days_late

    # Distance features for chronic lateness
    num_bills_30plus_days_late = np.zeros(n, dtype=int)
    mask_30plus = max_days_late >= 30
    num_bills_30plus_days_late[mask_30plus] = np.random.randint(1, 6, size=mask_30plus.sum())
    total_late_missed = num_payments_late_12m + missed_payments_12m
    num_bills_30plus_days_late = np.minimum(num_bills_30plus_days_late, total_late_missed)

    num_bills_60plus_days_late = np.zeros(n, dtype=int)
    mask_60plus = max_days_late >= 60
    num_bills_60plus_days_late[mask_60plus] = np.random.randint(1, 4, size=mask_60plus.sum())
    num_bills_60plus_days_late = np.minimum(num_bills_60plus_days_late, num_bills_30plus_days_late)

    days_since_last_payment = np.random.randint(0, 31, size=n)
    mask_last_missed = last_payment_status == "missed"
    days_since_last_payment[mask_last_missed] = np.random.randint(31, 121, size=mask_last_missed.sum())

    consecutive_missed_payments_max = np.zeros(n, dtype=int)
    mask_missed_some = missed_payments_12m > 0
    consecutive_missed_payments_max[mask_missed_some] = np.random.randint(
        1, 4, size=mask_missed_some.sum()
    )
    consecutive_missed_payments_max = np.minimum(
        consecutive_missed_payments_max, missed_payments_12m
    )

    current_outstanding_balance = np.random.uniform(0, 5000, size=n).round(2)
    current_outstanding_balance[missed_payments_12m > 0] = np.random.uniform(
        500, 50000, size=(missed_payments_12m > 0).sum()
    ).round(2)

    # --- BUILD DATAFRAME ---
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "plan_type": plan_types,
        "is_family_plan": is_family_plan,
        "avg_monthly_bill": avg_monthly_bill,
        "total_revenue_12m": total_revenue_12m,
        "late_payments_12m": late_payments_12m,
        "missed_payments_12m": missed_payments_12m,
        "max_days_late": max_days_late,
        "tenure_months": tenure_months,
        "num_bills": num_bills,
        "num_payments_on_time_12m": num_payments_on_time_12m,
        "num_payments_late_12m": num_payments_late_12m,
        "late_payment_ratio": late_payment_ratio,
        "missed_payment_ratio": missed_payment_ratio,
        "avg_payment_delay_days": avg_payment_delay_days,
        "last_payment_status": last_payment_status,
        "last_payment_delay_days": last_payment_delay_days,
        "days_overdue_current_bill": days_overdue_current_bill,
        "is_currently_overdue": is_currently_overdue,

        "payment_method": payment_method,
        "auto_pay_enabled": auto_pay_enabled,
        "billing_cycle_day": billing_cycle_day,
        "historical_billing_disputes": historical_billing_disputes,

        "customer_age": customer_age,
        "customer_region": customer_region,
        "occupation_type": occupation_type,
        "income_bracket": income_bracket,
        "account_type": account_type,
        "num_dependents": num_dependents,

        "avg_monthly_data_usage_gb": avg_monthly_data_usage_gb,
        "avg_monthly_voice_minutes": avg_monthly_voice_minutes,
        "avg_monthly_sms": avg_monthly_sms,
        "international_usage_minutes": international_usage_minutes,
        "roaming_usage_minutes": roaming_usage_minutes,
        "value_added_services_subscribed": value_added_services_subscribed,

        "plan_category": plan_category,
        "contract_length_months": contract_length_months,
        "discounts_or_offers_applied": discounts_or_offers_applied,
        "plan_change_count_last_12m": plan_change_count_last_12m,
        "is_5g_plan": is_5g_plan,
        "num_lines_in_account": num_lines_in_account,

        "num_support_calls_12m": num_support_calls_12m,
        "complaint_count_12m": complaint_count_12m,
        "issue_type_majority": issue_type_majority,
        "refund_requests_12m": refund_requests_12m,
        "dispute_count_12m": dispute_count_12m,

        "loyalty_score": loyalty_score,
        "app_login_frequency_monthly": app_login_frequency_monthly,
        "reward_points_balance": reward_points_balance,
        "renewal_rate": renewal_rate,

        "credit_score_range": credit_score_range,
        "wallet_balance": wallet_balance,
        "avg_topup_amount": avg_topup_amount,
        "topup_frequency": topup_frequency,
        "max_payment_delay_days_12m": max_payment_delay_days_12m,
        "num_bills_30plus_days_late": num_bills_30plus_days_late,
        "num_bills_60plus_days_late": num_bills_60plus_days_late,
        "days_since_last_payment": days_since_last_payment,
        "consecutive_missed_payments_max": consecutive_missed_payments_max,
        "current_outstanding_balance": current_outstanding_balance,
    })

    return df


def assign_segment_label(row):
    """
    Assign exactly THREE classes using a probabilistic rule:
      - 'habitual_defaulter'   -> high risk_score
      - 'occasional_defaulter' -> medium risk_score
      - 'critical'             -> high value_score (family / high revenue / loyal)
    Adds noise so model can't get fake 100% accuracy.
    """
    # ----- Risk score from payment behaviour -----
    m_norm = min(row["missed_payments_12m"] / 5.0, 1.0)
    l_norm = row["late_payment_ratio"]
    d_norm = min(row["max_days_late"] / 90.0, 1.0)
    overdue_norm = min(row["days_overdue_current_bill"] / 60.0, 1.0)

    risk_score = 0.4 * m_norm + 0.3 * l_norm + 0.2 * d_norm + 0.1 * overdue_norm
    risk_score = max(0.0, min(1.0, risk_score))

    # ----- Value score from revenue + tenure + lines + loyalty -----
    rev_norm = min(row["total_revenue_12m"] / 20000.0, 1.0)
    ten_norm = min(row["tenure_months"] / 60.0, 1.0)
    lines_norm = min(row["num_lines_in_account"] / 4.0, 1.0)
    loy_norm = row["loyalty_score"] / 10.0

    value_score = 0.45 * rev_norm + 0.25 * ten_norm + 0.15 * lines_norm + 0.15 * loy_norm
    value_score = max(0.0, min(1.0, value_score))

    # ----- Noisy logits -> softmax -----
    noise_h = np.random.normal(0, 0.25)
    noise_o = np.random.normal(0, 0.25)
    noise_c = np.random.normal(0, 0.25)

    log_habitual = 2.0 * risk_score - 0.5 + noise_h            # high risk
    log_critical = 2.0 * value_score - risk_score + noise_c     # high value, lower risk
    log_occasional = (1.0 - abs(risk_score - 0.5)
                      + 0.3 * value_score + noise_o)            # middle band

    logits = np.array([log_critical, log_habitual, log_occasional])
    exps = np.exp(logits - logits.max())
    probs = exps / exps.sum()

    classes = ["critical", "habitual_defaulter", "occasional_defaulter"]
    return np.random.choice(classes, p=probs)


if __name__ == "__main__":
    df = generate_telecom_data(N_CUSTOMERS)

    print("Assigning segment labels (3 categories, only postpaid)...")
    df["segment_label"] = df.apply(assign_segment_label, axis=1)

    print("\nClass Distribution:")
    print(df["segment_label"].value_counts(normalize=True))

    print("\nDataset Shape:", df.shape)

    # Train/test split + save
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=df["segment_label"]
    )

    df.to_csv("telecom_customers_full.csv", index=False, encoding="utf-8")
    train_df.to_csv("telecom_customers_train.csv", index=False, encoding="utf-8")
    test_df.to_csv("telecom_customers_test.csv", index=False, encoding="utf-8")

    print("\nSaved:")
    print("- telecom_customers_full.csv")
    print("- telecom_customers_train.csv")
    print("- telecom_customers_test.csv")
