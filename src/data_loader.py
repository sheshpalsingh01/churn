"""
data_loader.py
──────────────
Load raw data and — if no real CSV exists — generate a realistic
synthetic loan dataset so the project runs immediately.

Swap generate_sample_data() for your own data source.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.config import cfg


# ── Public API ────────────────────────────────────────────────────────────────

def load_raw_data(path: Path = None) -> pd.DataFrame:
    """
    Load raw CSV.  If the file doesn't exist, generate synthetic data,
    save it, and return it — so the pipeline works out of the box.
    """
    path = path or cfg.raw_data_path
    path = Path(path)

    if not path.exists():
        print(f"[data_loader] No file at {path}. Generating sample data...")
        df = generate_sample_data(n=1000, seed=cfg.random_seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"[data_loader] Sample data saved → {path}  shape={df.shape}")
    else:
        df = pd.read_csv(path)
        print(f"[data_loader] Loaded {path}  shape={df.shape}")

    _basic_validation(df)
    return df


# ── Sample data generator ─────────────────────────────────────────────────────

def generate_sample_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic loan approval dataset.
    Replace this with your actual data loading logic.
    """
    rng = np.random.default_rng(seed)

    gender          = rng.choice(["Male", "Female"], n, p=[0.80, 0.20])
    married         = rng.choice(["Yes", "No"],      n, p=[0.65, 0.35])
    dependents      = rng.choice(["0","1","2","3+"],  n, p=[0.57, 0.17, 0.16, 0.10])
    education       = rng.choice(["Graduate","Not Graduate"], n, p=[0.78, 0.22])
    self_employed   = rng.choice(["Yes", "No"],       n, p=[0.14, 0.86])
    property_area   = rng.choice(["Urban","Semiurban","Rural"], n, p=[0.38,0.34,0.28])
    credit_history  = rng.choice([1, 0], n, p=[0.84, 0.16]).astype(float)

    applicant_income    = rng.integers(1500,  15000, n).astype(float)
    coapplicant_income  = rng.integers(0,     6000,  n).astype(float)
    loan_amount         = rng.integers(50,    500,   n).astype(float)
    loan_term           = rng.choice([120, 180, 240, 300, 360, 480], n).astype(float)

    # Introduce realistic nulls
    _null_idx = lambda p: rng.random(n) < p
    credit_history[_null_idx(0.08)]   = np.nan
    loan_amount[_null_idx(0.03)]      = np.nan
    coapplicant_income[_null_idx(0.02)]= np.nan

    # Approval logic: income + credit history + loan size
    total_income = applicant_income + coapplicant_income
    score = (
        0.40 * (credit_history == 1).astype(float) +
        0.30 * (total_income / (loan_amount + 1e-3) > 10).astype(float) +
        0.15 * (education == "Graduate").astype(float) +
        0.15 * rng.random(n)
    )
    loan_status = np.where(score > 0.50, "Y", "N")

    df = pd.DataFrame({
        "gender"             : gender,
        "married"            : married,
        "dependents"         : dependents,
        "education"          : education,
        "self_employed"      : self_employed,
        "applicant_income"   : applicant_income,
        "coapplicant_income" : coapplicant_income,
        "loan_amount"        : loan_amount,
        "loan_term"          : loan_term,
        "credit_history"     : credit_history,
        "property_area"      : property_area,
        "loan_status"        : loan_status,
    })
    return df


# ── Validation ────────────────────────────────────────────────────────────────

def _basic_validation(df: pd.DataFrame):
    """Fail fast if data is obviously broken."""
    if df.empty:
        raise ValueError("[data_loader] Dataset is empty.")

    if cfg.target_col not in df.columns:
        raise ValueError(
            f"[data_loader] Target column '{cfg.target_col}' not found. "
            f"Available: {df.columns.tolist()}"
        )

    null_pct = df.isnull().mean()
    high_null = null_pct[null_pct > 0.50]
    if not high_null.empty:
        print(f"[data_loader] WARNING — columns >50% null: {high_null.to_dict()}")

    print(f"[data_loader] Columns   : {df.columns.tolist()}")
    print(f"[data_loader] Null count:\n{df.isnull().sum()[df.isnull().sum()>0]}")
    print(f"[data_loader] Target dist:\n{df[cfg.target_col].value_counts()}")
