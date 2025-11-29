**Project**: VIBE_CODERS — Synthetic Telecom Data Generator

- **Purpose**: Generates a realistic synthetic telecom-customer dataset for experimentation, model training, and testing.
- **Primary script**: `generate_telecom_data.py` — produces full, train, and test CSV files.

**Getting Started**
- **Prerequisites**: Python 3.8+ and packages listed in `requirements.txt`.
- **Install**: 

```powershell
python -m pip install -r requirements.txt
```

**Run generator**
- **Generate default dataset (10k rows)**:

```powershell
python generate_telecom_data.py
```

- **Change number of customers**: edit `N_CUSTOMERS` at top of `generate_telecom_data.py` or call the function from a REPL/script.

**Outputs**
- `telecom_customers_full.csv` — full generated dataset.
- `telecom_customers_train.csv` — 80% stratified train split by `segment_label`.
- `telecom_customers_test.csv` — 20% stratified test split.

**Key columns (selected)**
- **`customer_id`**: Unique ID.
- **`avg_monthly_bill`, `total_revenue_12m`**: Billing and revenue signals.
- **`missed_payments_12m`, `num_payments_late_12m`, `late_payment_ratio`**: Payment behavior metrics.
- **`max_days_late`, `num_bills_30plus_days_late`, `num_bills_60plus_days_late`**: Late-severity indicators.
- **`credit_score_range`, `wallet_balance`, `current_outstanding_balance`**: Financial stress indicators.
- **`loyalty_score`, `renewal_rate`, `reward_points_balance`**: Engagement/loyalty.
- **`segment_label`**: Assigned segment — one of `on_time_payer`, `occasional_defaulter`, `habitual_defaulter`, `critical`.

**Segment labeling logic (brief)**
- `habitual_defaulter`: `missed_payments_12m >= 3` or `late_payment_ratio > 0.5` or `max_days_late >= 60`.
- `occasional_defaulter`: `missed_payments_12m` in 1..2 or `0.2 <= late_payment_ratio <= 0.5`.
- `on_time_payer`: default when none of the above.
- `critical`: overrides other labels when customer meets high-value criteria (high revenue, long tenure, multi-line/family account, loyalty >=7, and non-poor credit).

**Reproducibility**
- The generator uses a fixed seed `RANDOM_SEED = 42` at the top of `generate_telecom_data.py`. Change the seed for different synthetic draws.

**Testing / Quick checks**
- After running, verify distribution and shape:

```powershell
python - <<'PY'
import pandas as pd
print(pd.read_csv('telecom_customers_full.csv').shape)
print(pd.read_csv('telecom_customers_full.csv')['segment_label'].value_counts(normalize=True))
PY
```

**Notes & caveats**
- This dataset is synthetic and intended for development/testing only. Do not use it to make production decisions without validating against real production data.
- Column distributions are simulated and may not capture all real-world edge cases.

**Next steps you might want**
- Add a CLI arg to `generate_telecom_data.py` to override `N_CUSTOMERS` and output paths.
- Add unit tests for label assignment logic in `assign_segment_label`.

**Contact / Maintainer**
- Repo owner: `MonishThelapalli` (local workspace owner semantics).

**License**
- Internal use only — add a license file if you plan to publish.
