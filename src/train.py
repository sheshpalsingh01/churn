"""
train.py
────────
Train the full sklearn Pipeline (preprocessor + model) and save
all artifacts: pipeline.joblib + metrics.json.

Run:   python -m src.train
  or:  make train
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.config import cfg
from src.data_loader import load_raw_data
from src.preprocess import preprocess, load_splits, splits_exist
from src.features import build_preprocessor, get_feature_columns, get_X_y
from src.evaluate import evaluate_and_save


# ── Model factory ─────────────────────────────────────────────────────────────

def _get_model():
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators    = cfg.n_estimators,
            max_depth       = cfg.max_depth,
            min_samples_leaf= cfg.min_samples_leaf,
            class_weight    = cfg.class_weight,
            random_state    = cfg.random_seed,
            n_jobs          = -1,
        ),
        "gradient_boost": GradientBoostingClassifier(
            n_estimators = cfg.n_estimators,
            max_depth    = cfg.max_depth,
            random_state = cfg.random_seed,
        ),
        "logistic": LogisticRegression(
            max_iter     = 1000,
            class_weight = cfg.class_weight,
            random_state = cfg.random_seed,
        ),
    }
    model = models.get(cfg.model_type)
    if model is None:
        raise ValueError(f"Unknown model_type '{cfg.model_type}'. Choose from {list(models)}")
    return model


# ── Main train function ───────────────────────────────────────────────────────

def train() -> Pipeline:
    cfg.ensure_dirs()

    # 1. Load / preprocess data
    if splits_exist():
        print("[train] Loading existing splits...")
        train_df, val_df, test_df = load_splits()
    else:
        print("[train] Preprocessing raw data...")
        raw = load_raw_data()
        train_df, val_df, test_df = preprocess(raw)

    # 2. X/y split
    X_train, y_train = get_X_y(train_df)
    X_val,   y_val   = get_X_y(val_df)
    X_test,  y_test  = get_X_y(test_df)

    # 3. Build preprocessor from training column types
    num_cols, cat_cols = get_feature_columns(train_df)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    # 4. Build full pipeline
    model = _get_model()
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model",        model),
    ])

    # 5. Cross-validation on training set
    print(f"\n[train] Running 5-fold CV with {cfg.model_type}...")
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv      = 5,
        scoring = "roc_auc",
        n_jobs  = -1,
    )
    print(f"[train] CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 6. Fit on full training data
    print("[train] Fitting on full train set...")
    pipeline.fit(X_train, y_train)

    # 7. Save pipeline
    joblib.dump(pipeline, cfg.pipeline_path)
    print(f"[train] Pipeline saved → {cfg.pipeline_path}")

    # 8. Evaluate on val + test, save metrics and plots
    metrics = evaluate_and_save(pipeline, X_val, y_val, X_test, y_test,
                                 cv_mean=cv_scores.mean(), cv_std=cv_scores.std())

    print(f"\n[train] ✓ Done.  Test ROC-AUC = {metrics['test_roc_auc']:.4f}")
    return pipeline


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
