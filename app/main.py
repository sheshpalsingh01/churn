"""
app/main.py
───────────
Streamlit entry point.
Run:  streamlit run app/main.py
"""

import sys
from pathlib import Path

# Make src importable from app/
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from src.config import cfg

st.set_page_config(
    page_title = cfg.app_title,
    page_icon  = "🏦",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

st.title(f"🏦 {cfg.app_title}")
st.markdown("""
Welcome! Use the sidebar to navigate between pages.

| Page | Description |
|------|-------------|
| **Predict** | Enter applicant details and get a live prediction |
| **Batch Predict** | Upload a CSV and predict for many applicants at once |
| **EDA** | Explore the training data with interactive charts |
| **Model Info** | View model performance metrics and feature importances |
""")

from src.predict import model_exists
if not model_exists():
    st.warning(
        "⚠️  No trained model found. "
        "Run `make train` in your terminal to train the model first."
    )
else:
    st.success("✅  Model loaded and ready.")
