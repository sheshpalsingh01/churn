"""
features.py
───────────
Build the sklearn ColumnTransformer that preprocesses features.
This transformer is fitted ONLY on training data, then saved as
part of the Pipeline — zero leakage, same transform in serving.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OrdinalEncoder,
    LabelEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline

from src.config import cfg


# ── Feature builder ──────────────────────────────────────────────────────────

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-specific features BEFORE passing to ColumnTransformer.
    These are deterministic transforms — no fitting needed.
    """
    df = df.copy()

    # Total income
    if "applicant_income" in df.columns and "coapplicant_income" in df.columns:
        df["total_income"] = df["applicant_income"] + df["coapplicant_income"]

    # EMI proxy: loan amount per month
    if "loan_amount" in df.columns and "loan_term" in df.columns:
        df["emi_proxy"] = df["loan_amount"] / (df["loan_term"] + 1e-3)

    # Loan to income ratio
    if "loan_amount" in df.columns and "total_income" in df.columns:
        df["loan_to_income"] = df["loan_amount"] / (df["total_income"] + 1e-3)

    return df


def get_feature_columns(df: pd.DataFrame) -> tuple[list, list]:
    """
    Separate numeric and categorical columns from a preprocessed df.
    Excludes the target column.
    """
    df = df.drop(columns=[cfg.target_col], errors="ignore")
    df = add_engineered_features(df)

    numeric    = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()

    return numeric, categorical


def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Returns an UNFITTED ColumnTransformer.
    Numeric: impute median → scale.
    Categorical: impute mode → ordinal encode.
    """
    numeric_transformer = SklearnPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_transformer = SklearnPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer,    numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",        # drop any column not listed
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_all_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return feature names after fit (for feature importance plots)."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []


# ── XY split helper ──────────────────────────────────────────────────────────

def get_X_y(df: pd.DataFrame):
    """
    Apply engineered features, split into X and y.
    Removes the target column from X.
    """
    df = add_engineered_features(df)
    X = df.drop(columns=[cfg.target_col])
    y = df[cfg.target_col]
    return X, y
