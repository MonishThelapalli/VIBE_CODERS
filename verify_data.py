import pandas as pd

try:
    df = pd.read_csv("telecom_customers_full.csv")
    print("Loaded telecom_customers_full.csv")
    
    # Check 1: Plan Type
    plan_types = df["plan_type"].unique()
    print(f"Unique plan types: {plan_types}")
    if len(plan_types) == 1 and plan_types[0] == "postpaid":
        print("✅ Plan type check passed: All are 'postpaid'.")
    else:
        print(f"❌ Plan type check failed: Found {plan_types}")
        
    # Check 2: Prepaid Columns
    columns = df.columns
    prepaid_cols = ["avg_topup_amount", "topup_frequency"]
    found_cols = [c for c in prepaid_cols if c in columns]
    if not found_cols:
        print("✅ Prepaid columns check passed: Columns removed.")
    else:
        print(f"❌ Prepaid columns check failed: Found {found_cols}")
        
    # Check 3: Class Distribution
    print("\nClass Distribution:")
    print(df["segment_label"].value_counts(normalize=True))
    
except Exception as e:
    print(f"❌ Verification failed with error: {e}")
