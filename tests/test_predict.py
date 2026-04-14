"""
tests/test_predict.py
─────────────────────
Tests for the train-predict pipeline end to end.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from src.data_loader import generate_sample_data
from src.preprocess import preprocess
from src.features import get_feature_columns, build_preprocessor, get_X_y
from src.config import cfg

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture(scope="module")
def trained_pipeline():
    """Build a small pipeline for testing — does not use saved artifacts."""
    raw = generate_sample_data(n=300, seed=42)
    train_df, _, _ = preprocess(raw)

    X_train, y_train = get_X_y(train_df)
    num_cols, cat_cols = get_feature_columns(train_df)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=10, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def test_pipeline_predicts(trained_pipeline):
    from src.data_loader import generate_sample_data
    raw = generate_sample_data(n=10, seed=7)
    raw = raw.drop(columns=[cfg.target_col])
    preds = trained_pipeline.predict(raw)
    assert len(preds) == 10


def test_predict_output_shape(trained_pipeline):
    from src.predict import predict
    sample_input = pd.DataFrame([{
        "gender": "Male", "married": "Yes", "dependents": "0",
        "education": "Graduate", "self_employed": "No",
        "applicant_income": 5000.0, "coapplicant_income": 0.0,
        "loan_amount": 150.0, "loan_term": 360.0,
        "credit_history": 1.0, "property_area": "Urban",
    }])
    result = predict(trained_pipeline, sample_input)
    assert "prediction"     in result
    assert "prob_approved"  in result
    assert "label"          in result
    assert result["prob_approved"] + result["prob_rejected"] == pytest.approx(1.0, abs=1e-4)


def test_batch_predict_columns(trained_pipeline):
    from src.predict import predict_batch
    raw = generate_sample_data(n=20, seed=3)
    raw = raw.drop(columns=[cfg.target_col])
    result = predict_batch(trained_pipeline, raw)
    for col in ["prediction", "label", "prob_approved", "confidence"]:
        assert col in result.columns
    assert len(result) == 20
