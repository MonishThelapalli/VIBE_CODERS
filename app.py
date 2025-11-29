# app.py
import streamlit as st
import pandas as pd

from classifier import classify_customer
from email_generator import generate_email_for_row


st.set_page_config(page_title="Customer Payment Classification & Communication Assistant", layout="wide")

st.title("📨 Customer Payment Classification & Communication Assistant (Demo)")

st.markdown(
    """
Upload a CSV in the **TelecomCustomerChurn** format.  
For each customer, the system will:

1. Predict their segment:
   - `critical`
   - `occasional_defaulter`
   - `habitual_defaulter`
2. Generate a category-specific email using predefined templates (simulating an LLM).
"""
)

uploaded_file = st.file_uploader("Upload Telecom customer CSV", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Raw Metadata Preview")
    st.dataframe(df.head())

    # Ensure required columns exist
    required_cols = {"customerID", "Tenure", "MonthlyCharges", "TotalCharges", "Churn"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"CSV is missing required columns: {', '.join(sorted(missing))}")
    else:
        # Add category column using our "ML model"
        df["category"] = df.apply(classify_customer, axis=1)

        st.subheader("📊 Segment Distribution")
        st.bar_chart(df["category"].value_counts())

        st.subheader("📧 Generated Emails (simulated LLM using templates)")
        max_rows = st.slider(
            "How many customers to display?", min_value=1, max_value=min(50, len(df)), value=min(10, len(df))
        )

        # Optionally filter by category
        category_filter = st.multiselect(
            "Filter by category (optional)", options=df["category"].unique().tolist()
        )

        display_df = df.copy()
        if category_filter:
            display_df = display_df[display_df["category"].isin(category_filter)]

        display_df = display_df.head(max_rows)

        for idx, row in display_df.iterrows():
            category = row["category"]
            subject, body = generate_email_for_row(row, category)

            header = f"{row['customerID']}  |  Segment: {category}"
            with st.expander(header):
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.markdown("**Customer Metadata (from CSV)**")
                    st.write(
                        {
                            "customerID": row["customerID"],
                            "Tenure": int(row["Tenure"]),
                            "MonthlyCharges": float(row["MonthlyCharges"]),
                            "TotalCharges": row["TotalCharges"],
                            "Churn": row["Churn"],
                        }
                    )

                with col2:
                    st.markdown("**Email Preview**")
                    st.markdown(f"**Subject:** {subject}")
                    st.text(body)

        st.success("Emails generated based on the rule-based model and templates.")
else:
    st.info("Upload a CSV file to get started.")
