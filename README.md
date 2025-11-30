# VIBE_CODERS — Customer Payment Classification & Communication Assistant (Hackathon README)

**Purpose**: Generate synthetic telecom customer data, preprocess it, train segmentation/classification models with leak-aware safeguards, and provide a small API and utilities for inference and templated outreach.

This README documents how to run the project end-to-end, explains the role of each file in the repository, highlights important design choices (leakage control, fairness), and gives quick commands for common tasks.

**Quick Hackathon Pitch**
- **What**: A complete pipeline that goes from synthetic data generation → preprocessing → model training (multiple algorithms, XGBoost with leakage detection) → saved model artifact → FastAPI prediction endpoint → templated outreach generator.
- **Why**: Useful for experimenting with payment risk segmentation, building targeted communication flows, and evaluating fairness/leakage risks before production.

**Repo layout (high-level)**
- `generate_telecom_data.py`: Synthetic dataset generator (default 10k rows). Produces `telecom_customers_full.csv` and splits.
- `preprocess_telecom_data.py`: Data quality checks, preprocessing pipeline builder, and saves `processed_data/preprocessor.joblib`, `label_encoder.joblib`, `train.csv`, `test.csv`.
- `train_payment_classifier.py`: Trains multiple classifiers (Logistic, RandomForest, GradientBoosting), selects best model, saves `best_model_pipeline.joblib`.
- `train_xgboost.py`: Advanced XGBoost training with automatic leakage detection and fairness exclusions; saves `xgb_customer_payment_classifier.joblib`.
- `prediction_pipeline.py`: Helpers to load artifacts, preprocess single rows, and produce predictions plus probabilities.
- `app.py`: FastAPI app exposing `GET /predict/{customer_id}` to return features + prediction.
- `data_loader.py`: Simple CSV loader that fetches a single customer row by `customer_id` and returns expected feature order.
- `email_generator.py` + `templates_db.py`: Simple templated outreach generator using customer metadata and segmentation to build subject/body pairs.
- `processed_data/`: Artifacts and processed CSVs created by preprocessing.
- Other scripts: `verify_data.py`, `qt.py`, `classifier.py`, `models.py` (small helpers/tests), and `requirements.txt`.

**Prerequisites**
- Python 3.8+ (3.10 recommended)
- PowerShell on Windows (commands below assume PowerShell)
- Install packages:

```powershell
python -m pip install -r requirements.txt
```

If you prefer only core packages for running generation and preprocessing:

```powershell
python -m pip install numpy pandas scikit-learn joblib
```

**End-to-end workflow (recommended)**

1. Generate synthetic data

```powershell
python generate_telecom_data.py
```

This will produce at repo root:
- `telecom_customers_full.csv`
- `telecom_customers_train.csv`
- `telecom_customers_test.csv`

2. Preprocess data (quality checks, build preprocessor)

```powershell
python preprocess_telecom_data.py
```

Artifacts written to `processed_data/`:
- `preprocessor.joblib` — the ColumnTransformer or pipeline used to transform features
- `label_encoder.joblib` — LabelEncoder for `segment_label` → numeric
- `train.csv`, `test.csv` — processed CSVs with `target` column

3. Train classification models (baseline and best-pick)

```powershell
python train_payment_classifier.py
```

This runs several models, prints metrics, and saves `best_model_pipeline.joblib` (pipeline combining preprocessor + best classifier).

4. (Optional) Train XGBoost with leakage control

```powershell
python train_xgboost.py
```

This script performs automatic leakage detection (Mutual Information, correlation, perfect predictors), excludes sensitive demographic columns for basic fairness, runs hyperparameter tuning, saves `xgb_customer_payment_classifier.joblib`, and writes a confusion matrix image.

5. Start the prediction API

```powershell
python app.py
```

Then query the API (example):

```powershell
curl http://127.0.0.1:8000/predict/C0001
```

6. Generate templated email for a row (example, interactive or script)

Use `email_generator.py` to turn a customer row and predicted category into subject/body pairs using templates in `templates_db.py`.

**Files and responsibilities (detailed)**
- `generate_telecom_data.py` — Generates realistic features (billing, payments, tenure, usage, demographics, engagement) and assigns `segment_label` via `assign_segment_label` logic (on-time / occasional / habitual / critical). It also produces stratified train/test splits.
- `preprocess_telecom_data.py` — Runs QA checks (missing values, duplicates, logical rules), builds a `ColumnTransformer` that standardizes numeric features and one-hot encodes categoricals, encodes the target, splits into train/test, and saves artifacts to `processed_data/`.
- `train_payment_classifier.py` — Loads processed artifacts, transforms data, trains Logistic Regression, Random Forest, and a tuned Gradient Boosting model, evaluates them, and writes the best pipeline to `best_model_pipeline.joblib`.
- `train_xgboost.py` — A more advanced training script with:
	- Automatic leakage detection using Mutual Information (MI), numeric correlation, and perfect-predictor checks.
	- A conservative manual exclusion list for known leakage features (e.g., `missed_payments_12m`, `late_payment_ratio`, `current_outstanding_balance`, etc.).
	- Sensitive column exclusions for fairness (e.g., `customer_age`, `income_bracket`).
	- Builds an XGBoost pipeline, tunes hyperparameters with `RandomizedSearchCV`, evaluates and saves the model and confusion matrix.
- `prediction_pipeline.py` — Utilities to load model & preprocessor, validate/reorder a single-row DataFrame to expected features, transform it, and run `predict` / `predict_proba`. Returns human-readable labels using a saved label encoder.
- `app.py` — FastAPI wrapper exposing `/predict/{customer_id}`. On startup it loads the model and preprocessor from artifacts, uses `DataLoader` to fetch a single-row, preprocesses, predicts, and returns JSON with features and predicted label/probabilities.
- `data_loader.py` — CSV-based loader with helpers:
	- `get_customer_by_id(customer_id)` returns a one-row DataFrame.
	- `get_expected_feature_columns()` reads `processed_data/train.csv` header to know the exact feature order used during training (important for single-row transforms).
- `email_generator.py` + `templates_db.py` — A rules-based templating system to create outreach content depending on segment. Useful to demo how segmentation can drive personalized messaging.
- `templates_db.py` — Template text for `critical`, `occasional_defaulter`, and `habitual_defaulter` categories with placeholders.
- `qt.py` / `verify_data.py` / `classifier.py` — Small helper or exploratory scripts included for quick checks and prototyping. `verify_data.py` performs a few sanity assertions on `telecom_customers_full.csv`.
- `processed_data/` — Directory where preprocessing artifacts and processed CSVs are stored after running `preprocess_telecom_data.py`.
- `best_model_pipeline.joblib`, `xgb_customer_payment_classifier.joblib` — Example saved models / artifacts (some already present in the repo).

**Key design notes / reasoning**
- Leakage prevention: Many payment-related columns directly reveal the label (missed/late payments). `train_xgboost.py` purposely detects and removes these features and also includes a manual blacklist so the model must rely on safer behavioral and historical signals.
- Fairness: The pipeline demonstrates a simple step to exclude sensitive demographics from training — include more advanced fairness checks (e.g., group metrics) as needed.
- Reproducibility: Synthetic generator uses a fixed seed by default (`RANDOM_SEED = 42`) so runs are repeatable unless you change the seed.

**Quick troubleshooting**
- `FileNotFoundError` for `processed_data/*`: run `preprocess_telecom_data.py` first.
- FastAPI startup errors about model/preprocessor: ensure `best_model_pipeline.joblib` and `processed_data/preprocessor.joblib` exist and are readable.
- Stratify errors during train/test split: ensure `segment_label` exists and isn't extremely imbalanced for the chosen split.

**Suggested improvements for hackathon demo**
- Add CLI flags to `generate_telecom_data.py` (`--count`, `--seed`, `--out-dir`) and to training scripts (`--model-out`, `--skip-tune`).
- Add a small Streamlit dashboard to visualize class distributions and model performance (quick prototype).
- Add `pytest` tests covering `assign_segment_label` and `prediction_pipeline.preprocess_input_row`.
- Add a CI workflow (GitHub Actions) to run tests and a lint pass on push.

**Commands summary**
- Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

- Generate data:

```powershell
python generate_telecom_data.py
```

- Preprocess data:

```powershell
python preprocess_telecom_data.py
```

- Train models (baseline):

```powershell
python train_payment_classifier.py
```

- Train XGBoost (leakage-aware):

```powershell
python train_xgboost.py
```

- Run prediction API:

```powershell
python app.py
curl http://127.0.0.1:8000/predict/C0001
```

**License & Attribution**
- No license included in this repository. For hackathon delivery, add a short permissive license (e.g., MIT) if you plan to publish.

**Contact / Maintainers**
- Repo owner (local workspace): `MonishThelapalli`.

If you want, I can now:
- Add CLI flags to `generate_telecom_data.py` and `train_*` scripts and run them to verify outputs.
- Add `tests/test_labeling.py` with `pytest` and create a simple GitHub Actions workflow.
- Create a short Streamlit dashboard (`app_ui.py`) that calls the FastAPI endpoint and visualizes predictions.

Pick one and I'll implement it next.
