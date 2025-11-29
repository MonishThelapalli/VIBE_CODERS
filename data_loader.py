import os
from typing import Optional, List

import pandas as pd


class DataLoader:
    """Simple loader to fetch customer rows by id from a CSV file.

    Features:
    - Supports `customer_id` and `customerID` column names.
    - Provides `get_expected_feature_columns()` which reads the header of processed_data/train.csv
      to determine the exact feature order used during training.
    """

    def __init__(self, data_file: str = None):
        if data_file is None:
            data_file = os.path.join(os.path.dirname(__file__), "telecom_customers_full.csv")
        self.data_file = data_file
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        self._df = None

    def _ensure_loaded(self):
        if self._df is None:
            self._df = pd.read_csv(self.data_file)

    def get_customer_by_id(self, customer_id: str) -> Optional[pd.DataFrame]:
        """Return a single-row DataFrame for the given customer_id. Supports both `customer_id` and `customerID`."""
        self._ensure_loaded()
        df = self._df

        # try common column names
        id_cols = [c for c in ["customer_id", "customerID", "customerId", "customerID"] if c in df.columns]
        if not id_cols:
            # fallback: if there is a column with 'customer' in name
            cand = [c for c in df.columns if 'customer' in c.lower()]
            id_cols = cand[:1]

        if not id_cols:
            return None

        id_col = id_cols[0]
        match = df[df[id_col].astype(str) == str(customer_id)]
        if match.empty:
            return None

        # Return one-row DataFrame (keep original column names)
        return match.iloc[[0]]

    def get_expected_feature_columns(self) -> List[str]:
        """Return list of expected feature column names (in order) used during training.

        It reads `processed_data/train.csv` header and excludes the `target` column.
        """
        processed_train = os.path.join(os.path.dirname(__file__), "processed_data", "train.csv")
        if not os.path.exists(processed_train):
            raise FileNotFoundError(f"Processed train.csv not found: {processed_train}")

        # read only header
        df_head = pd.read_csv(processed_train, nrows=0)
        cols = list(df_head.columns)
        # drop 'target' if present
        cols = [c for c in cols if c != 'target']
        return cols
