"""
tests/test_preprocess.py
────────────────────────
Unit tests for data loading and preprocessing.
Run: pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest

from src.data_loader import generate_sample_data
from src.preprocess import preprocess
from src.config import cfg


@pytest.fixture
def raw_df():
    return generate_sample_data(n=200, seed=0)


def test_sample_data_shape(raw_df):
    assert raw_df.shape[0] == 200
    assert cfg.target_col in raw_df.columns


def test_sample_data_has_target(raw_df):
    assert set(raw_df[cfg.target_col].unique()).issubset({"Y", "N"})


def test_preprocess_returns_three_splits(raw_df):
    train, val, test = preprocess(raw_df)
    assert len(train) > 0
    assert len(val)   > 0
    assert len(test)  > 0


def test_no_nulls_after_preprocess(raw_df):
    train, val, test = preprocess(raw_df)
    for split in [train, val, test]:
        assert split.isnull().sum().sum() == 0, "Nulls remain after preprocessing"


def test_target_is_binary_after_preprocess(raw_df):
    train, _, _ = preprocess(raw_df)
    assert set(train[cfg.target_col].unique()).issubset({0, 1})


def test_no_data_leakage(raw_df):
    """Train and test indices must not overlap."""
    train, val, test = preprocess(raw_df)
    train_idx = set(train.index)
    val_idx   = set(val.index)
    test_idx  = set(test.index)
    assert len(train_idx & test_idx) == 0, "Train/test overlap!"
    assert len(train_idx & val_idx)  == 0, "Train/val overlap!"
    assert len(val_idx   & test_idx) == 0, "Val/test overlap!"
