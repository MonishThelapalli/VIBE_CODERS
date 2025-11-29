import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# Configuration
INPUT_FILE = "telecom_customers_full.csv"
OUTPUT_DIR = "processed_data"
PREPROCESSOR_FILE = "preprocessor.joblib"

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Data loaded. Shape: {df.shape}")
    return df

def perform_quality_checks(df):
    print("\n--- Data Quality & Sanity Checks ---")
    
    # 1. Missing Values
    print("\n1. Missing Values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("✅ No missing values found.")
    else:
        print(missing)
        
    # 2. Duplicate Rows
    print("\n2. Duplicate Rows:")
    n_dupes = df["customer_id"].duplicated().sum()
    if n_dupes == 0:
        print("✅ No duplicate customer_ids found.")
    else:
        print(f"⚠️ Found {n_dupes} duplicate customer_ids.")
        
    # 3. Target Distribution
    print("\n3. Target Distribution:")
    print(df["segment_label"].value_counts(normalize=True))
    
    # 4. Basic Numeric Stats
    print("\n4. Basic Numeric Stats:")
    print(df.describe().T[["min", "max", "mean", "std"]])
    
    # 5. Consistency Checks
    print("\n5. Consistency Checks:")
    
    # Rule 1: Payments <= Bills
    rule1_violations = df[
        (df["num_payments_on_time_12m"] + df["num_payments_late_12m"]) > df["num_bills"]
    ]
    print(f"Rule 1: num_payments (on_time + late) <= num_bills — violated in {len(rule1_violations)} rows")
    
    # Rule 2: Late Payment Ratio [0, 1]
    rule2_violations = df[
        (df["late_payment_ratio"] < 0) | (df["late_payment_ratio"] > 1)
    ]
    print(f"Rule 2: late_payment_ratio between 0 and 1 — violated in {len(rule2_violations)} rows")
    
    # Rule 3: Missed Payment Ratio [0, 1]
    rule3_violations = df[
        (df["missed_payment_ratio"] < 0) | (df["missed_payment_ratio"] > 1)
    ]
    print(f"Rule 3: missed_payment_ratio between 0 and 1 — violated in {len(rule3_violations)} rows")
    
    # Rule 4: Overdue Logic (Positive)
    rule4_violations = df[
        (df["days_overdue_current_bill"] > 0) & (df["is_currently_overdue"] != 1)
    ]
    print(f"Rule 4: If days_overdue > 0 then is_currently_overdue == 1 — violated in {len(rule4_violations)} rows")
    
    # Rule 5: Overdue Logic (Zero)
    rule5_violations = df[
        (df["days_overdue_current_bill"] == 0) & (df["is_currently_overdue"] != 0)
    ]
    print(f"Rule 5: If days_overdue == 0 then is_currently_overdue == 0 — violated in {len(rule5_violations)} rows")
    
    # Rule 6: Tenure > 0
    rule6_violations = df[df["tenure_months"] <= 0]
    print(f"Rule 6: tenure_months > 0 — violated in {len(rule6_violations)} rows")
    
    # Rule 7: Non-negative financials
    rule7_violations = df[
        (df["avg_monthly_bill"] < 0) | (df["total_revenue_12m"] < 0)
    ]
    print(f"Rule 7: avg_monthly_bill and total_revenue_12m >= 0 — violated in {len(rule7_violations)} rows")

def preprocess_and_split(df):
    print("\n--- Preprocessing & Splitting ---")
    
    # Define Column Groups
    target_col = "segment_label"
    id_col = "customer_id"
    
    # Categorical Features
    cat_cols = [
        "plan_type", "last_payment_status", "payment_method", 
        "customer_region", "occupation_type", "income_bracket", 
        "account_type", "plan_category", "issue_type_majority", 
        "credit_score_range"
    ]
    
    # Binary Features (Treat as numeric/passthrough or ordinal)
    # Since they are 0/1, we can just pass them through.
    bin_cols = [
        "is_family_plan", "is_currently_overdue", "auto_pay_enabled", 
        "value_added_services_subscribed", "discounts_or_offers_applied", 
        "is_5g_plan"
    ]
    
    # Numeric Features
    # All columns except target, id, cat, and bin
    all_cols = df.columns.tolist()
    exclude_cols = [target_col, id_col] + cat_cols + bin_cols
    num_cols = [c for c in all_cols if c not in exclude_cols]
    
    print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
    print(f"Binary columns ({len(bin_cols)}): {bin_cols}")
    print(f"Numeric columns ({len(num_cols)}): {num_cols}")
    
    # Split X and y
    X = df.drop(columns=[target_col, id_col])
    y = df[target_col]
    
    # Encode Target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Target classes: {le.classes_}")
    
    # Build Preprocessor
    # Numeric: Standard Scaler
    # Categorical: OneHotEncoder
    # Binary: Passthrough
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
            ('bin', 'passthrough', bin_cols)
        ],
        verbose_feature_names_out=False
    )
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Fit Preprocessor on Train
    print("Fitting preprocessor...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"Processed Train shape: {X_train_processed.shape}")
    
    # Save Artifacts
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    joblib.dump(preprocessor, os.path.join(OUTPUT_DIR, PREPROCESSOR_FILE))
    joblib.dump(le, os.path.join(OUTPUT_DIR, "label_encoder.joblib"))
    
    # Save processed arrays (optional, but useful)
    # Or save the split raw dataframes
    train_df = X_train.copy()
    train_df["target"] = y_train
    test_df = X_test.copy()
    test_df["target"] = y_test
    
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)
    
    print(f"Artifacts saved to {OUTPUT_DIR}/")
    print(f"- {PREPROCESSOR_FILE}")
    print(f"- label_encoder.joblib")
    print(f"- train.csv")
    print(f"- test.csv")

if __name__ == "__main__":
    df = load_data(INPUT_FILE)
    
    # Print basic info
    print("\n--- Basic Info ---")
    print(df.info())
    print(df.head())
    
    perform_quality_checks(df)
    preprocess_and_split(df)
