import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
from src.predict import load_model, predict_batch, model_exists
from src.config import cfg

st.set_page_config(page_title="Batch Predict", page_icon="📂", layout="wide")
st.title("📂 Batch Prediction")
st.caption("Upload a CSV file with applicant data. Get predictions for all rows at once.")


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_model()


if not model_exists():
    st.error("No trained model found. Run `make train` first.")
    st.stop()

model = get_model()

# ── Upload ────────────────────────────────────────────────────────────────────
st.info(
    "CSV must have columns: `gender, married, dependents, education, "
    "self_employed, applicant_income, coapplicant_income, "
    "loan_amount, loan_term, credit_history, property_area`"
)

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write(f"Loaded **{len(df)} rows** · {df.shape[1]} columns")

    with st.expander("Preview raw data"):
        st.dataframe(df.head(10), use_container_width=True)

    if st.button("Run Batch Prediction", type="primary"):
        with st.spinner("Predicting..."):
            result_df = predict_batch(model, df)

        st.success(f"Done! {len(result_df)} predictions made.")

        # Summary
        col1, col2, col3 = st.columns(3)
        approved_count = (result_df["prediction"] == 1).sum()
        col1.metric("Total applicants", len(result_df))
        col2.metric("Approved",  int(approved_count),
                    delta=f"{approved_count/len(result_df):.1%}")
        col3.metric("Rejected",  int(len(result_df) - approved_count))

        # Full result table
        st.dataframe(
            result_df[["label", "prob_approved", "confidence"] +
                       [c for c in result_df.columns if c not in
                        ["label","prob_approved","confidence","prediction"]]],
            use_container_width=True,
        )

        # Download
        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label    = "⬇️  Download results as CSV",
            data     = csv_bytes,
            file_name= "predictions.csv",
            mime     = "text/csv",
        )
else:
    # Sample download
    st.subheader("Don't have a file? Download a sample template:")
    sample = pd.DataFrame([{
        "gender": "Male", "married": "Yes", "dependents": "0",
        "education": "Graduate", "self_employed": "No",
        "applicant_income": 5000, "coapplicant_income": 1500,
        "loan_amount": 150, "loan_term": 360,
        "credit_history": 1, "property_area": "Urban",
    }])
    st.download_button(
        "⬇️  Download sample template",
        sample.to_csv(index=False).encode("utf-8"),
        file_name="sample_template.csv",
        mime="text/csv",
    )
