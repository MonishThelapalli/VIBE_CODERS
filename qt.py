import pickle
import pandas as pd

# Load your trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the dataset
df = pd.read_csv("telecom_customers_full.csv")

# Fetch customer_id and segment_label
customer_ids = df["customer_id"]
segment_labels = df["segment_label"]

print(customer_ids.head())
print(segment_labels.head())
