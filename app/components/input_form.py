"""
app/components/input_form.py
─────────────────────────────
Reusable Streamlit input form for single prediction.
Returns a dict of raw feature values matching the model's input schema.
"""

import streamlit as st
import pandas as pd


def render_input_form() -> pd.DataFrame | None:
    """
    Render all input fields.
    Returns a one-row DataFrame ready for predict(), or None if not submitted.
    """
    with st.form("loan_input_form"):
        st.subheader("Applicant Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])

        with col2:
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        with col3:
            credit_history = st.selectbox(
                "Credit History",
                [1, 0],
                format_func=lambda x: "Good (1)" if x == 1 else "Bad (0)"
            )

        st.subheader("Financial Details")
        col4, col5 = st.columns(2)

        with col4:
            applicant_income = st.number_input(
                "Applicant Monthly Income (₹)",
                min_value=0, max_value=200000,
                value=5000, step=500,
            )
            coapplicant_income = st.number_input(
                "Co-applicant Income (₹)",
                min_value=0, max_value=100000,
                value=0, step=500,
            )

        with col5:
            loan_amount = st.number_input(
                "Loan Amount (₹ thousands)",
                min_value=10, max_value=1000,
                value=150, step=10,
            )
            loan_term = st.selectbox(
                "Loan Term (months)",
                [60, 120, 180, 240, 300, 360, 480],
                index=5,
            )

        submitted = st.form_submit_button("🔍  Predict", use_container_width=True)

    if not submitted:
        return None

    data = {
        "gender"            : gender,
        "married"           : married,
        "dependents"        : dependents,
        "education"         : education,
        "self_employed"     : self_employed,
        "applicant_income"  : float(applicant_income),
        "coapplicant_income": float(coapplicant_income),
        "loan_amount"       : float(loan_amount),
        "loan_term"         : float(loan_term),
        "credit_history"    : float(credit_history),
        "property_area"     : property_area,
    }
    return pd.DataFrame([data])
