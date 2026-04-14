"""
predict.py
──────────
Load the trained pipeline and run predictions.
Streamlit imports get_model() and predict() from here.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import cfg
from src.features import add_engineered_features


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model():
    """Load pipeline from disk. Raises FileNotFoundError if not trained yet."""
    if not cfg.pipeline_path.exists():
        raise FileNotFoundError(
            f"No trained model at {cfg.pipeline_path}.\n"
            "Run `make train` or `python -m src.train` first."
        )
    pipeline = joblib.load(cfg.pipeline_path)
    return pipeline


def model_exists() -> bool:
    return cfg.pipeline_path.exists()


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(model, input_df: pd.DataFrame) -> dict:
    """
    Run inference on a DataFrame of raw features.
    Returns dict with prediction, probability, and label.
    """
    df = add_engineered_features(input_df.copy())

    # Drop target column if accidentally included
    df = df.drop(columns=[cfg.target_col], errors="ignore")

    raw_pred  = model.predict(df)
    probas    = model.predict_proba(df)

    results = {
        "prediction"    : int(raw_pred[0]),
        "label"         : "Approved ✅" if raw_pred[0] == 1 else "Rejected ❌",
        "confidence"    : float(probas[0].max()),
        "prob_approved" : float(probas[0][1]),
        "prob_rejected" : float(probas[0][0]),
        "high_confidence": float(probas[0].max()) >= cfg.confidence_threshold,
    }
    return results


def predict_batch(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Batch prediction on a DataFrame.
    Returns original df with new columns: prediction, label, prob_approved.
    """
    df_feat = add_engineered_features(df.copy())
    df_feat = df_feat.drop(columns=[cfg.target_col], errors="ignore")

    preds  = model.predict(df_feat)
    probas = model.predict_proba(df_feat)

    result = df.copy()
    result["prediction"]  = preds
    result["label"]       = np.where(preds == 1, "Approved", "Rejected")
    result["prob_approved"] = np.round(probas[:, 1], 4)
    result["confidence"]  = np.round(probas.max(axis=1), 4)

    return result
