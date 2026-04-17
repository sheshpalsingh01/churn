# Loan Approval Predictor

A machine learning application that predicts loan approval status based on applicant information using a Streamlit web interface.

## 🚀 Features

- **Interactive Web App**: User-friendly Streamlit interface for loan prediction
- **Machine Learning Models**: Multiple ML algorithms (Logistic Regression, Random Forest, Gradient Boosting, XGBoost)
- **Data Processing**: Comprehensive data cleaning and preprocessing pipeline
- **Feature Engineering**: Automated feature creation and transformation
- **Model Evaluation**: Cross-validation and performance metrics
- **Visualization**: Charts and analytics for decision factors

## 📋 Requirements

- Python 3.8+
- Dependencies listed in `requirement.txt`

## 🛠 Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd churn
   ```

2. **Create and activate virtual environment:**
   ```bash
   conda create -n loan-env python=3.10
   conda activate loan-env
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirement.txt
   ```

## 📊 Data Preparation

1. **Process raw data:**
   ```bash
   python src/processor.py
   ```

2. **Feature engineering:**
   ```bash
   python src/feature_engineering.py
   ```

3. **Train models:**
   ```bash
   python src/train_model.py
   ```

## 🌐 Usage

### Run the Web Application

```bash
cd app
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Input Features

- Gender
- Marital Status
- Number of Dependents
- Education Level
- Self-Employment Status
- Property Area
- Credit History
- Applicant Income
- Co-applicant Income
- Loan Amount
- Loan Term

## 📁 Project Structure

```
churn/
├── app/
│   ├── app.py              # Streamlit web application
│   └── config.toml         # Streamlit configuration
├── data/
│   ├── raw/
│   │   └── loan_data.csv   # Raw dataset
│   └── processed/
│       ├── cleaned_loan_data.csv
│       └── featured.csv    # Engineered features
├── model/
│   └── trained/
│       └── best_model.pkl  # Trained model
├── models/
│   └── preprocessor.joblib # Fitted preprocessor
├── notebook/
│   ├── 01_data_engineering.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_experementation.ipynb
├── src/
│   ├── feature_engineering.py
│   ├── processor.py
│   └── train_model.py
├── requirement.txt          # Python dependencies
└── README.md
```

## 🤖 Model Details

### Algorithms Used
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier (if available)

### Features Engineered
- Total Income (Applicant + Co-applicant)
- Income to Loan Ratio
- Monthly Loan Amount
- Income Share Ratio
- High Income Flag

### Preprocessing
- One-Hot Encoding for categorical variables
- Standard Scaling for numerical variables
- Missing value imputation

## 🚀 Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Connect your repository to [Streamlit Cloud](https://share.streamlit.io/)
3. Deploy directly from the platform

### Local Deployment

```bash
streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0
```

## 📈 Model Performance

The model selection process compares multiple algorithms and selects the best performing one based on accuracy score. The trained model is saved as `best_model.pkl` in the `model/trained/` directory.
