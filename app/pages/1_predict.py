import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from src.predict import load_model, predict, model_exists
from src.config import cfg
from app.components.input_form import render_input_form
from app.components.charts import probability_gauge

st.set_page_config(page_title="Predict", page_icon="🔍", layout="wide")
st.title("🔍 Single Prediction")
st.caption("Fill in the applicant details and click Predict.")


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_model()


if not model_exists():
    st.error("No trained model found. Run `make train` first.")
    st.stop()

model = get_model()

# ── Form ──────────────────────────────────────────────────────────────────────
input_df = render_input_form()

if input_df is not None:
    result = predict(model, input_df)

    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.metric("Decision", result["label"])
        if result["high_confidence"]:
            st.success(f"High confidence: {result['confidence']:.1%}")
        else:
            st.warning(f"Low confidence: {result['confidence']:.1%} — review manually")

    with col2:
        st.metric("Approval Probability", f"{result['prob_approved']:.1%}")
        st.metric("Rejection Probability", f"{result['prob_rejected']:.1%}")

    with col3:
        st.plotly_chart(
            probability_gauge(result["prob_approved"]),
            use_container_width=True,
        )

    with st.expander("📋 Input data submitted"):
        st.dataframe(input_df, use_container_width=True)
