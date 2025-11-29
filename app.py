"""
FastAPI server for prediction of customer payment segment.

GET /predict/{customer_id} -> returns full customer features + predicted label

This file is a lightweight FastAPI wrapper that uses the prediction pipeline
artifacts in `processed_data/` and the trained model `best_model_pipeline.joblib`.

Notes:
- Uses `prediction_pipeline.py` helpers: load_model_and_preprocessor(), preprocess_input_row(), run_model_prediction()
- Returns JSON and includes error handling when a customer_id is not found.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, Optional
import os

from prediction_pipeline import (
    load_model_and_preprocessor,
    preprocess_input_row,
    run_model_prediction,
)
from data_loader import DataLoader

# Config - artifact locations (relative to repository root)
MODEL_FILE = os.path.join(os.path.dirname(__file__), "best_model_pipeline.joblib")
PREPROCESSOR_FILE = os.path.join(os.path.dirname(__file__), "processed_data", "preprocessor.joblib")
LABEL_ENCODER_FILE = os.path.join(os.path.dirname(__file__), "processed_data", "label_encoder.joblib")
DATA_FILE = os.path.join(os.path.dirname(__file__), "telecom_customers_full.csv")

app = FastAPI(title="Telecom Customer Segment Prediction API")


class PredictionResponse(BaseModel):
    customer_id: str
    features: Dict[str, Any]
    predicted_label: str
    predicted_proba: Optional[Dict[str, float]] = None


# Load artifacts once on startup
MODEL_ARTIFACT = None
PREPROCESSOR = None
CLASSIFIER = None
LABEL_ENCODER = None


@app.on_event("startup")
def startup_event():
    global MODEL_ARTIFACT, PREPROCESSOR, CLASSIFIER, LABEL_ENCODER
    MODEL_ARTIFACT, PREPROCESSOR, CLASSIFIER, LABEL_ENCODER = load_model_and_preprocessor(
        model_path=MODEL_FILE,
        preprocessor_path=PREPROCESSOR_FILE,
        label_encoder_path=LABEL_ENCODER_FILE,
    )


@app.get("/predict/{customer_id}", response_model=PredictionResponse)
def predict_customer(customer_id: str):
    # Load customer row
    dl = DataLoader(DATA_FILE)
    row = dl.get_customer_by_id(customer_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"customer_id '{customer_id}' not found")

    # Preprocess and predict
    try:
        # Ensure expected features/order
        expected_cols = dl.get_expected_feature_columns()
        X_proc_df = preprocess_input_row(row, expected_cols, PREPROCESSOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {e}")

    try:
        pred_label, pred_proba = run_model_prediction(CLASSIFIER, LABEL_ENCODER, X_proc_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    resp = PredictionResponse(
        customer_id=customer_id,
        features=row.to_dict(orient="records")[0],
        predicted_label=pred_label,
        predicted_proba=pred_proba,
    )

    return JSONResponse(status_code=200, content=resp.dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
