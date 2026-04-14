"""
utils.py
────────
Shared helpers used across the project.
"""

import json
from pathlib import Path
from src.config import cfg


def load_metrics() -> dict | None:
    """Load saved metrics.json. Returns None if not trained yet."""
    if not cfg.metrics_path.exists():
        return None
    with open(cfg.metrics_path) as f:
        return json.load(f)


def fmt_pct(value: float) -> str:
    """Format float as percentage string: 0.947 → '94.7%'"""
    return f"{value * 100:.1f}%"


def fmt_metric(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"
