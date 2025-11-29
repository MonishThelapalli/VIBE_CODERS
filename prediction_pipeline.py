import os
from typing import Tuple, List, Dict, Optional

import joblib
import numpy as np
import pandas as pd


def load_model_and_preprocessor(model_path: str, preprocessor_path: str, label_encoder_path: str):
    """Load model artifact, preprocessor, and label encoder.

    Returns: (model_artifact, preprocessor, classifier, label_encoder)
    classifier may be the same as model_artifact or extracted from a saved Pipeline.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Label encoder file not found: {label_encoder_path}")

    model_artifact = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    label_encoder = joblib.load(label_encoder_path)

    # Extract classifier if model_artifact is a sklearn Pipeline
    classifier = model_artifact
    try:
        # sklearn Pipeline stores pipeline.steps as list of (name, estimator)
        steps = getattr(model_artifact, "steps", None)
        if steps and isinstance(steps, list):
            # assume last step is the classifier
            classifier = steps[-1][1]
    except Exception:
        classifier = model_artifact

    return model_artifact, preprocessor, classifier, label_encoder


def preprocess_input_row(row_df: pd.DataFrame, expected_columns: List[str], preprocessor) -> np.ndarray:
    """Validate, reorder, and transform a single-row DataFrame using provided preprocessor.

    - row_df: pandas DataFrame with one row
    - expected_columns: list of column names expected by preprocessor (order matters)
    - preprocessor: fitted transformer with .transform()

    Returns: numpy array (1, n_features_transformed)
    """
    if not isinstance(row_df, pd.DataFrame):
        raise ValueError("row_df must be a pandas DataFrame with a single row")
    if row_df.shape[0] != 1:
        raise ValueError("row_df must contain exactly one row")

    # Ensure all expected cols present; if missing fill with sensible default (0 or empty string)
    missing = [c for c in expected_columns if c not in row_df.columns]
    if missing:
        # Fill missing with zeros or empty strings depending on dtype guess
        for c in missing:
            row_df[c] = 0

    # Reorder
    X = row_df[expected_columns].copy()

    # Some preprocessors require proper dtypes - try to let pandas infer
    # Transform
    X_transformed = preprocessor.transform(X)

    return X_transformed


def run_model_prediction(classifier, label_encoder, X_transformed: np.ndarray) -> Tuple[str, Optional[Dict[str, float]]]:
    """Run classifier.predict and return human-readable label and probability mapping.

    - classifier: trained classifier with predict and optionally predict_proba
    - label_encoder: sklearn LabelEncoder fitted during training (to invert labels)
    - X_transformed: numpy array returned by preprocessor.transform
    """
    # Predict
    pred = classifier.predict(X_transformed)

    # pred may be array-like
    pred_arr = np.array(pred).ravel()
    # Map back to label string
    try:
        pred_label = label_encoder.inverse_transform(pred_arr.astype(int))[0]
    except Exception:
        # fallback to string
        pred_label = str(pred_arr[0])

    proba_dict = None
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(X_transformed)
        probs = np.array(probs)
        probs = probs.ravel() if probs.shape[0] == 1 else probs[0]

        classes = getattr(classifier, "classes_", None)
        if classes is not None:
            # try to translate classes to label names using label_encoder
            try:
                class_names = label_encoder.inverse_transform(classes.astype(int))
            except Exception:
                # classes might already be label strings
                class_names = [str(c) for c in classes]
        else:
            # No classes_ - map using label_encoder.classes_
            class_names = list(label_encoder.classes_)

        proba_dict = {str(name): float(p) for name, p in zip(class_names, probs)}

    return pred_label, proba_dict
