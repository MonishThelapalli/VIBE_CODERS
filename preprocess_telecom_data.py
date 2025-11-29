import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Constants
N_CUSTOMERS = 10000
RANDOM_SEED = 42

def generate_telecom_data(n=N_CUSTOMERS):
    """
    Generates a synthetic telecom customer dataset with n rows.
    """
    np.random.seed(RANDOM_SEED)
    
    print(f"Generating data for {n} customers...")
    
    # --- 1. BASIC TELECOM INFO ---
    customer_ids = [f"C{str(i).zfill(4)}" for i in range(1, n + 1)]
    
    # Plan type: 60% prepaid, 40% postpaid
    plan_types = np.random.choice(["prepaid", "postpaid"], size=n, p=[0.6, 0.4])
    
    # Family plan: 20% yes
    is_family_plan = np.random.choice([0, 1], size=n, p=[0.8, 0.2])
    
    # Avg monthly bill: 200 - 3000
    # Use a log-normal distribution to simulate real spending (skewed right) but clipped to range
    avg_monthly_bill = np.random.lognormal(mean=6.5, sigma=0.6, size=n)
    avg_monthly_bill = np.clip(avg_monthly_bill, 200, 3000).round(2)
    
    # Tenure months: 1 - 120
    tenure_months = np.random.randint(1, 121, size=n)
    
    # Total revenue 12m: avg_bill * min(12, tenure) + noise
    months_active_12m = np.minimum(12, tenure_months)
    noise = np.random.normal(0, 50, size=n)
    total_revenue_12m = (avg_monthly_bill * months_active_12m) + noise
    total_revenue_12m = np.maximum(0, total_revenue_12m).round(2) # Ensure non-negative
    
    # Late & Missed payments (Base probabilities)
    # We'll adjust these later for specific segments, but start with a distribution
    # Most people pay on time.
    late_payments_12m = np.random.poisson(lam=1.5, size=n) # Average 1.5 late payments
    late_payments_12m = np.minimum(late_payments_12m, 12)
    
    missed_payments_12m = np.random.poisson(lam=0.5, size=n) # Average 0.5 missed
    missed_payments_12m = np.minimum(missed_payments_12m, 8)
    
    # Ensure missed <= late (usually if you miss, it was late first, or count logic)
    # But here they are separate counts. Let's keep them somewhat independent but correlated.
    
    # Max days late: correlated with late/missed
    max_days_late = np.zeros(n, dtype=int)
    mask_late = late_payments_12m > 0
    max_days_late[mask_late] = np.random.randint(1, 30, size=np.sum(mask_late))
    
    mask_missed = missed_payments_12m > 0
    # If missed, max days late is likely higher (30+)
    max_days_late[mask_missed] = np.random.randint(30, 121, size=np.sum(mask_missed))
    
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "plan_type": plan_types,
        "is_family_plan": is_family_plan,
        "avg_monthly_bill": avg_monthly_bill,
        "total_revenue_12m": total_revenue_12m,
        "late_payments_12m": late_payments_12m,
        "missed_payments_12m": missed_payments_12m,
        "max_days_late": max_days_late,
        "tenure_months": tenure_months
    })

    # --- 2. PAYMENT TIMING & RELIABILITY ---
    # Assume total_payments_12m approx min(12, tenure)
    total_payments_12m = np.minimum(12, df["tenure_months"])
    
    # Fix late/missed to not exceed total payments
    df["late_payments_12m"] = np.minimum(df["late_payments_12m"], total_payments_12m)
    df["missed_payments_12m"] = np.minimum(df["missed_payments_12m"], total_payments_12m)
    
    # On time = Total - Late - Missed? 
    # Usually "Late" implies paid but late. "Missed" implies not paid.
    # Let's assume: Total opportunities = 12 (or tenure).
    # On Time = Total - (Late + Missed). 
    # But sometimes Late includes Missed in definitions. Let's assume mutually exclusive for sum check:
    # Actually, let's say: 
    # Late = Paid but late.
    # Missed = Not paid.
    # On Time = Paid on time.
    # So Sum = Total.
    
    # Adjust counts to sum to total_payments_12m
    # We prioritize missed, then late, then on_time
    df["missed_payments_12m"] = np.minimum(df["missed_payments_12m"], total_payments_12m)
    remaining = total_payments_12m - df["missed_payments_12m"]
    df["late_payments_12m"] = np.minimum(df["late_payments_12m"], remaining)
    df["num_payments_on_time_12m"] = total_payments_12m - df["missed_payments_12m"] - df["late_payments_12m"]
    
    # Renaming for clarity with prompt
    df["num_payments_late_12m"] = df["late_payments_12m"]
    
    # Ratios
    # Avoid division by zero
    safe_total = total_payments_12m.replace(0, 1) 
    df["late_payment_ratio"] = (df["num_payments_late_12m"] / safe_total).round(2)
    df["missed_payment_ratio"] = (df["missed_payments_12m"] / safe_total).round(2)
    
    # Avg payment delay days
    # If on time, delay is 0. If late, say 5-29 days. If missed, say 30+ (but technically undefined or high)
    # We'll simulate an average based on the counts.
    avg_delay = np.zeros(n)
    # Contribution from late
    avg_delay += (df["num_payments_late_12m"] * np.random.uniform(5, 25, size=n))
    # Contribution from missed (effectively very late)
    avg_delay += (df["missed_payments_12m"] * np.random.uniform(30, 60, size=n))
    df["avg_payment_delay_days"] = (avg_delay / safe_total).round(1)
    
    # Last payment info
    df["last_payment_status"] = np.random.choice(["on_time", "late", "missed"], size=n, p=[0.7, 0.2, 0.1])
    # Correlate last status with general behavior
    mask_habitual = (df["missed_payments_12m"] > 2)
    df.loc[mask_habitual, "last_payment_status"] = np.random.choice(["late", "missed"], size=sum(mask_habitual), p=[0.4, 0.6])
    
    df["last_payment_delay_days"] = 0
    df.loc[df["last_payment_status"] == "late", "last_payment_delay_days"] = np.random.randint(1, 30, size=sum(df["last_payment_status"] == "late"))
    df.loc[df["last_payment_status"] == "missed", "last_payment_delay_days"] = np.random.randint(30, 90, size=sum(df["last_payment_status"] == "missed"))

    # --- 3. BILLING & PAYMENT METHOD ---
    df["payment_method"] = np.random.choice(["bank_transfer", "credit_card", "UPI", "cash"], size=n)
    
    # Auto pay: correlated with good behavior
    df["auto_pay_enabled"] = np.random.choice([0, 1], size=n, p=[0.6, 0.4])
    # Boost auto pay for those with 0 missed payments
    mask_good = df["missed_payments_12m"] == 0
    df.loc[mask_good, "auto_pay_enabled"] = np.random.choice([0, 1], size=sum(mask_good), p=[0.4, 0.6])
    
    df["billing_cycle_day"] = np.random.randint(1, 31, size=n)
    df["historical_billing_disputes"] = np.random.poisson(0.5, size=n)
    
    # --- 4. CUSTOMER PROFILE & DEMOGRAPHICS ---
    df["customer_age"] = np.random.randint(18, 76, size=n)
    df["customer_region"] = np.random.choice(["north", "south", "east", "west", "metro", "rural"], size=n)
    df["occupation_type"] = np.random.choice(["student", "salaried", "self_employed", "unemployed", "retired"], size=n, p=[0.1, 0.4, 0.3, 0.05, 0.15])
    df["income_bracket"] = np.random.choice(["low", "lower_middle", "upper_middle", "high"], size=n, p=[0.2, 0.4, 0.3, 0.1])
    df["account_type"] = np.random.choice(["individual", "SME", "corporate"], size=n, p=[0.85, 0.10, 0.05])
    df["num_dependents"] = np.random.randint(0, 6, size=n)
    
    # --- 5. TELECOM USAGE BEHAVIOR ---
    # Correlate with plan type and bill
    base_usage_gb = df["avg_monthly_bill"] / 20 # rough proxy
    df["avg_monthly_data_usage_gb"] = np.random.normal(base_usage_gb, 10, size=n).clip(1, 200).round(1)
    
    df["avg_monthly_voice_minutes"] = np.random.randint(50, 3001, size=n)
    df["avg_monthly_sms"] = np.random.randint(0, 1001, size=n)
    df["international_usage_minutes"] = np.random.exponential(10, size=n).clip(0, 500).round(0)
    df["roaming_usage_minutes"] = np.random.exponential(20, size=n).clip(0, 1000).round(0)
    df["value_added_services_subscribed"] = np.random.choice([0, 1], size=n, p=[0.7, 0.3])

    # --- 6. SUBSCRIPTION & PLAN METADATA ---
    df["plan_category"] = np.random.choice(["basic", "standard", "premium"], size=n, p=[0.4, 0.4, 0.2])
    # Adjust plan category based on bill
    df.loc[df["avg_monthly_bill"] > 1500, "plan_category"] = "premium"
    
    df["contract_length_months"] = np.random.choice([1, 3, 6, 12, 24], size=n)
    df["discounts_or_offers_applied"] = np.random.choice([0, 1], size=n, p=[0.6, 0.4])
    df["plan_change_count_last_12m"] = np.random.poisson(0.3, size=n).clip(0, 5)
    df["is_5g_plan"] = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
    
    df["num_lines_in_account"] = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.7, 0.15, 0.08, 0.05, 0.02])
    # Family plans have more lines
    df.loc[df["is_family_plan"] == 1, "num_lines_in_account"] = np.random.choice([2, 3, 4, 5], size=sum(df["is_family_plan"] == 1))
    
    # --- 7. CUSTOMER SUPPORT & COMPLAINT HISTORY ---
    df["num_support_calls_12m"] = np.random.poisson(2, size=n)
    df["complaint_count_12m"] = np.random.poisson(0.5, size=n)
    df["issue_type_majority"] = np.random.choice(["billing", "network", "other"], size=n)
    df["refund_requests_12m"] = np.random.poisson(0.1, size=n)
    df["dispute_count_12m"] = np.random.poisson(0.1, size=n)
    
    # Increase complaints for those with billing disputes or late payments
    mask_issues = (df["historical_billing_disputes"] > 0) | (df["missed_payments_12m"] > 0)
    df.loc[mask_issues, "complaint_count_12m"] += np.random.randint(1, 5, size=sum(mask_issues))
    df.loc[mask_issues, "issue_type_majority"] = "billing"

    # --- 8. ENGAGEMENT & LOYALTY ---
    df["loyalty_score"] = np.random.randint(1, 11, size=n)
    # Higher loyalty for longer tenure
    df.loc[df["tenure_months"] > 24, "loyalty_score"] = np.random.randint(5, 11, size=sum(df["tenure_months"] > 24))
    
    df["app_login_frequency_monthly"] = np.random.randint(0, 61, size=n)
    df["reward_points_balance"] = np.random.randint(0, 10001, size=n) * df["loyalty_score"]
    df["renewal_rate"] = np.random.uniform(0.1, 1.0, size=n).round(2)
    
    # --- 9. FINANCIAL STRESS INDICATORS ---
    df["credit_score_range"] = np.random.choice(["good", "fair", "poor"], size=n, p=[0.4, 0.4, 0.2])
    # Correlate with missed payments
    df.loc[df["missed_payments_12m"] >= 2, "credit_score_range"] = "poor"
    
    df["wallet_balance"] = np.random.uniform(0, 5000, size=n).round(2)
    df["avg_topup_amount"] = np.random.uniform(50, 2000, size=n).round(2)
    df["topup_frequency"] = np.random.randint(0, 31, size=n)
    
    # --- 10. EXTRA PAYMENT DELAY FEATURES ---
    df["max_payment_delay_days_12m"] = df["max_days_late"] # Consistency
    
    df["num_bills_30plus_days_late"] = np.zeros(n, dtype=int)
    # Approximate based on max_days_late and late counts
    mask_30plus = df["max_days_late"] >= 30
    df.loc[mask_30plus, "num_bills_30plus_days_late"] = np.random.randint(1, 6, size=sum(mask_30plus))
    # Ensure it doesn't exceed total late+missed
    total_late_missed = df["num_payments_late_12m"] + df["missed_payments_12m"]
    df["num_bills_30plus_days_late"] = np.minimum(df["num_bills_30plus_days_late"], total_late_missed)
    
    df["num_bills_60plus_days_late"] = np.zeros(n, dtype=int)
    mask_60plus = df["max_days_late"] >= 60
    df.loc[mask_60plus, "num_bills_60plus_days_late"] = np.random.randint(1, 4, size=sum(mask_60plus))
    df["num_bills_60plus_days_late"] = np.minimum(df["num_bills_60plus_days_late"], df["num_bills_30plus_days_late"])
    
    df["days_since_last_payment"] = np.random.randint(0, 31, size=n)
    # If last status missed, days since last payment is high
    mask_last_missed = df["last_payment_status"] == "missed"
    df.loc[mask_last_missed, "days_since_last_payment"] = np.random.randint(31, 121, size=sum(mask_last_missed))
    
    df["consecutive_missed_payments_max"] = np.zeros(n, dtype=int)
    mask_missed_some = df["missed_payments_12m"] > 0
    df.loc[mask_missed_some, "consecutive_missed_payments_max"] = np.random.randint(1, 4, size=sum(mask_missed_some))
    # Cap at missed count
    df["consecutive_missed_payments_max"] = np.minimum(df["consecutive_missed_payments_max"], df["missed_payments_12m"])
    
    df["current_outstanding_balance"] = np.random.uniform(0, 500, size=n).round(2)
    # Higher balance for defaulters
    mask_defaulters = df["missed_payments_12m"] > 0
    df.loc[mask_defaulters, "current_outstanding_balance"] = np.random.uniform(500, 50000, size=sum(mask_defaulters)).round(2)

    return df

def assign_segment_label(row):
    """
    Assigns a segment label based on the provided logic.
    """
    # Step 1: Define payment risk category
    base_risk = "on_time_payer"
    
    if (row["missed_payments_12m"] >= 3) or (row["late_payment_ratio"] > 0.5) or (row["max_days_late"] >= 60):
        base_risk = "habitual_defaulter"
    elif (row["missed_payments_12m"] in [1, 2]) or (0.2 <= row["late_payment_ratio"] <= 0.5):
        base_risk = "occasional_defaulter"
    
    segment_label = base_risk
    
    # Step 3: Override label to "critical" for high-value customers
    # Criteria:
    # - total_revenue_12m > 12000
    # - tenure_months > 12
    # - is_family_plan == 1 OR num_lines_in_account >= 3
    # - loyalty_score >= 7
    # - credit_score_range is not "poor"
    
    is_high_rev = row["total_revenue_12m"] > 12000
    is_long_tenure = row["tenure_months"] > 12
    is_multi_line = (row["is_family_plan"] == 1) or (row["num_lines_in_account"] >= 3)
    is_loyal = row["loyalty_score"] >= 7
    good_credit = row["credit_score_range"] != "poor"
    
    if is_high_rev and is_long_tenure and is_multi_line and is_loyal and good_credit:
        segment_label = "critical"
        
    return segment_label

if __name__ == "__main__":
    # 1. Generate Data
    df = generate_telecom_data(N_CUSTOMERS)
    
    # 2. Assign Labels
    print("Assigning segment labels...")
    df["segment_label"] = df.apply(assign_segment_label, axis=1)
    
    # 3. Sanity Checks
    print("\nClass Distribution:")
    print(df["segment_label"].value_counts(normalize=True))
    
    print("\nDataset Shape:", df.shape)
    
    # 4. Train/Test Split
    print("\nSplitting data into train (80%) and test (20%)...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["segment_label"])
    
    # 5. Save Files
    print("Saving CSV files...")
    df.to_csv("telecom_customers_full.csv", index=False, encoding='utf-8')
    train_df.to_csv("telecom_customers_train.csv", index=False, encoding='utf-8')
    test_df.to_csv("telecom_customers_test.csv", index=False, encoding='utf-8')
    
    print("Done! Files saved:")
    print("- telecom_customers_full.csv")
    print("- telecom_customers_train.csv")
    print("- telecom_customers_test.csv")
