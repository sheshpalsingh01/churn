"""
preprocess.py
─────────────
Clean raw data → train / val / test splits.
All splits saved to data/processed/ for reproducibility.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.config import cfg


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline.
    Returns (train_df, val_df, test_df) — each includes target column.
    """
    cfg.ensure_dirs()

    df = _clean(df)
    train_df, val_df, test_df = _split(df)

    # Save splits
    train_df.to_csv(cfg.processed_dir / "train.csv", index=False)
    val_df.to_csv(cfg.processed_dir  / "val.csv",   index=False)
    test_df.to_csv(cfg.processed_dir / "test.csv",  index=False)

    print(f"[preprocess] train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    return train_df, val_df, test_df


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pre-saved splits (skip reprocessing on re-runs)."""
    train = pd.read_csv(cfg.processed_dir / "train.csv")
    val   = pd.read_csv(cfg.processed_dir / "val.csv")
    test  = pd.read_csv(cfg.processed_dir / "test.csv")
    return train, val, test


def splits_exist() -> bool:
    return all(
        (cfg.processed_dir / f).exists()
        for f in ["train.csv", "val.csv", "test.csv"]
    )


# ── Cleaning ──────────────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Lowercase column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # 2. Drop duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"[preprocess] Dropped {before - len(df)} duplicate rows.")

    # 3. Fill numeric nulls with median (computed per column)
    for col in cfg.numeric_cols:
        if col in df.columns and df[col].isnull().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)
            print(f"[preprocess] Filled null in '{col}' with median={median:.2f}")

    # 4. Fill categorical nulls with mode
    for col in cfg.categorical_cols:
        if col in df.columns and df[col].isnull().any():
            mode = df[col].mode()[0]
            df[col] = df[col].fillna(mode)
            print(f"[preprocess] Filled null in '{col}' with mode='{mode}'")

    # 5. Strip whitespace in string columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())

    # 6. Encode target to binary int (Y→1, N→0)
    # Use try/cast — handles both classic object dtype and newer pandas StringDtype
    try:
        df[cfg.target_col] = df[cfg.target_col].astype(int)
    except (ValueError, TypeError):
        df[cfg.target_col] = (df[cfg.target_col] == "Y").astype(int)

    # 7. Remove any remaining rows with nulls
    before = len(df)
    df = df.dropna()
    print(f"[preprocess] Dropped {before - len(df)} rows still containing nulls.")

    print(f"[preprocess] Clean shape: {df.shape}")
    return df


# ── Splitting ─────────────────────────────────────────────────────────────────

def _split(df: pd.DataFrame):
    """
    Stratified split: train / val / test
    Sizes controlled by cfg.test_size and cfg.val_size.
    """
    y = df[cfg.target_col]

    # Step 1: hold out test
    train_val, test = train_test_split(
        df,
        test_size    = cfg.test_size,
        stratify     = y,
        random_state = cfg.random_seed,
    )

    # Step 2: carve val from train_val
    val_ratio = cfg.val_size / (1 - cfg.test_size)
    train, val = train_test_split(
        train_val,
        test_size    = val_ratio,
        stratify     = train_val[cfg.target_col],
        random_state = cfg.random_seed,
    )
    return train, val, test
