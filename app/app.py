
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / 'model' / 'trained' / 'best_model.pkl'
PREPROCESSOR_PATH = PROJECT_ROOT / 'models' / 'preprocessor.joblib'


@st.cache_resource
def load_artifacts():
    """Load trained model and fitted preprocessor."""
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor

# Page config
st.set_page_config(
    page_title='Loan Approval Predictor',
    page_icon='💳',
    layout='wide'
)

try:
    model, preprocessor = load_artifacts()
except FileNotFoundError as error:
    st.error(
        "Model files are missing. Run feature engineering and training first."
    )
    st.caption(str(error))
    st.stop()
except AttributeError as error:
    st.error(
        "The saved model files were created with a different scikit-learn "
        "version. Regenerate them in the same environment used to run this app."
    )
    st.code(
        "python src/feature_engineering.py\n"
        "python src/train_model.py",
        language="bash"
    )
    st.caption(str(error))
    st.stop()


def create_input_data(
    gender,
    married,
    dependents,
    education,
    self_employed,
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_term,
    credit_history,
    property_area
):
    """Build the same feature set used during training."""
    input_data = pd.DataFrame({
        'gender': [gender],
        'married': [married],
        'dependents': [dependents],
        'education': [education],
        'self_employed': [self_employed],
        'applicant_income': [applicant_income],
        'coapplicant_income': [coapplicant_income],
        'loan_amount': [loan_amount],
        'loan_term': [loan_term],
        'credit_history': [credit_history],
        'property_area': [property_area]
    })

    input_data['total_income'] = (
        input_data['applicant_income'] + input_data['coapplicant_income']
    )
    input_data['income_loan_ratio'] = (
        input_data['total_income'] / (input_data['loan_amount'] + 1)
    )
    input_data['monthly_loan_amount'] = (
        input_data['loan_amount'] / (input_data['loan_term'] + 1)
    )
    input_data['income_share_ratio'] = (
        input_data['applicant_income'] / (input_data['total_income'] + 1)
    )
    input_data['high_income_flag'] = np.where(
        input_data['total_income'] > 7000,
        1,
        0
    )

    return input_data


def predict_loan(input_data):
    transformed_input = pd.DataFrame(
        preprocessor.transform(input_data),
        columns=preprocessor.get_feature_names_out()
    )

    prediction = model.predict(transformed_input)[0]
    probability = (
        model.predict_proba(transformed_input)[0][1]
        if hasattr(model, 'predict_proba')
        else 0.0
    )

    return prediction, probability


def render_charts(input_data, probability, credit_history, loan_amount):
    total_income = float(input_data['total_income'][0])
    applicant_income = float(input_data['applicant_income'][0])
    coapplicant_income = float(input_data['coapplicant_income'][0])

    income_mix = pd.DataFrame(
        {
            'Amount': [applicant_income, coapplicant_income]
        },
        index=['Applicant Income', 'Co-applicant Income']
    )

    affordability = pd.DataFrame(
        {
            'Value': [
                total_income,
                float(input_data['loan_amount'][0]),
                float(input_data['monthly_loan_amount'][0])
            ]
        },
        index=['Total Income', 'Loan Amount', 'Monthly Loan Amount']
    )

    signal_scores = pd.DataFrame(
        {
            'Score': [
                round(probability * 100, 2),
                100 if credit_history == 1 else 20,
                100 if total_income > 7000 else 55,
                max(10, min(100, 100 - loan_amount / 5))
            ]
        },
        index=[
            'Approval Probability',
            'Credit History',
            'Income Strength',
            'Loan Size Comfort'
        ]
    )

    chart_col1, chart_col2, chart_col3 = st.columns(3)

    with chart_col1:
        st.markdown('#### Income Mix')
        st.bar_chart(income_mix)

    with chart_col2:
        st.markdown('#### Financial Snapshot')
        st.bar_chart(affordability)

    with chart_col3:
        st.markdown('#### Decision Signals')
        st.bar_chart(signal_scores)


# Custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background: #ffffff;
        color: #202331;
    }

    .block-container {
        max-width: 1060px;
        padding-top: 4rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        color: #202331;
        letter-spacing: 0;
    }

    [data-testid="stForm"] {
        border: 1px solid #d6d8de;
        border-radius: 8px;
        padding: 24px 14px 14px;
        background: #ffffff;
    }

    [data-testid="stSelectbox"] label,
    [data-testid="stNumberInput"] label {
        color: #202331;
        font-size: 13px;
    }

    .stSelectbox div[data-baseweb="select"] > div,
    .stNumberInput input {
        background: #eef0f4;
        border: 0;
        border-radius: 8px;
        color: #202331;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        border: 1px solid #d6d8de;
        border-radius: 8px;
        background: #ffffff;
        color: #202331;
        height: 38px;
    }

    .result-card {
        border: 1px solid #d6d8de;
        border-radius: 8px;
        padding: 18px;
        background: #ffffff;
        margin-top: 18px;
    }

    .approved {
        background: #eaf7ee;
        border: 1px solid #38a169;
        color: #166534;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        font-size: 22px;
        font-weight: 700;
    }

    .rejected {
        background: #fff1f2;
        border: 1px solid #e11d48;
        color: #9f1239;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        font-size: 22px;
        font-weight: 700;
    }

    [data-testid="stMetric"] {
        border: 1px solid #d6d8de;
        border-radius: 8px;
        padding: 12px;
        background: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Loan Approval Predictor")

with st.form('loan_prediction_form'):
    st.markdown('## Applicant Details')

    applicant_col1, applicant_col2, applicant_col3 = st.columns(3)

    with applicant_col1:
        gender = st.selectbox('Gender', ['Male', 'Female'])
        married = st.selectbox('Married', ['Yes', 'No'])
        dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])

    with applicant_col2:
        education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
        self_employed = st.selectbox('Self Employed', ['No', 'Yes'])
        property_area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])

    with applicant_col3:
        credit_history = st.selectbox(
            'Credit History',
            [1, 0],
            format_func=lambda value: 'Good (1)' if value == 1 else 'Bad (0)'
        )

    st.markdown('## Financial Details')

    finance_col1, finance_col2 = st.columns(2)

    with finance_col1:
        applicant_income = st.number_input(
            'Applicant Monthly Income (₹)',
            min_value=0.0,
            value=5000.0,
            step=100.0
        )
        coapplicant_income = st.number_input(
            'Co-applicant Income (₹)',
            min_value=0.0,
            value=0.0,
            step=100.0
        )

    with finance_col2:
        loan_amount = st.number_input(
            'Loan Amount (₹ thousands)',
            min_value=0.0,
            value=150.0,
            step=10.0
        )
        loan_term = st.selectbox(
            'Loan Term (months)',
            [12, 36, 60, 84, 120, 180, 240, 300, 360],
            index=8
        )

    predict_button = st.form_submit_button('🔍 Predict', use_container_width=True)

if predict_button:
    input_data = create_input_data(
        gender=gender,
        married=married,
        dependents=dependents,
        education=education,
        self_employed=self_employed,
        applicant_income=applicant_income,
        coapplicant_income=coapplicant_income,
        loan_amount=loan_amount,
        loan_term=loan_term,
        credit_history=credit_history,
        property_area=property_area
    )

    prediction, probability = predict_loan(input_data)

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown('## Prediction Result')

    if prediction == 1:
        st.markdown('<div class="approved">Loan Approved</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="rejected">Loan Rejected</div>', unsafe_allow_html=True)

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric('Approval Probability', f'{probability * 100:.2f}%')

    with metric_col2:
        st.metric('Total Income', f'₹{input_data["total_income"][0]:,.0f}')

    with metric_col3:
        st.metric(
            'Income / Loan Ratio',
            f'{input_data["income_loan_ratio"][0]:.2f}'
        )

    st.progress(float(probability))

    st.markdown('### Visual Analysis')
    render_charts(input_data, probability, credit_history, loan_amount)

    st.markdown('### Key Decision Factors')

    factors = []

    if credit_history == 1:
        factors.append('Good credit history increased approval chances')
    else:
        factors.append('Poor credit history reduced approval chances')

    if input_data['total_income'][0] > 7000:
        factors.append('Strong total income improved eligibility')

    if loan_amount > 250:
        factors.append('Higher loan amount increased lending risk')

    if property_area == 'Semiurban':
        factors.append('Semiurban property area slightly improved approval pattern')

    if not factors:
        factors.append('The model found a balanced applicant profile')

    for factor in factors:
        st.write(f'• {factor}')

    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info('Fill in the applicant details and click Predict.')
