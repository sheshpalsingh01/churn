
import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('feature-engineering')


def create_features(df):
    """Create new features from existing loan dataset."""
    logger.info("Creating new features")
     
    # Make a copy
    df_featured = df.copy()

    # Total income feature
    df_featured['total_income'] = (
        df_featured['applicant_income'] + df_featured['coapplicant_income']
    )
    logger.info("Created 'total_income' feature")

    # Income to loan ratio
    df_featured['income_loan_ratio'] = (
        df_featured['total_income'] / (df_featured['loan_amount'] + 1)
    )
    logger.info("Created 'income_loan_ratio' feature")

    # EMI approximation
    df_featured['monthly_loan_amount'] = (
        df_featured['loan_amount'] / (df_featured['loan_term'] + 1)
    )
    logger.info("Created 'monthly_loan_amount' feature")

    # Applicant vs Coapplicant contribution ratio
    df_featured['income_share_ratio'] = (
        df_featured['applicant_income'] / (df_featured['total_income'] + 1)
    )
    logger.info("Created 'income_share_ratio' feature")

    # High income flag
    df_featured['high_income_flag'] = np.where(
        df_featured['total_income'] > df_featured['total_income'].median(),
        1,
        0
    )
    logger.info("Created 'high_income_flag' feature")

    return df_featured


def create_preprocessor(categorical_features, numerical_features):
    """Create preprocessing pipeline."""
    logger.info("Creating preprocessing pipeline")

    # Numerical pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combined preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


def run_feature_engineering(input_file, output_file, preprocessor_file):
    """Full feature engineering pipeline."""

    # Load cleaned data
    logger.info(f"Loading cleaned data from {input_file}")
    df = pd.read_csv(input_file)

    # Create engineered features
    df_featured = create_features(df)
    logger.info(f"Feature engineered dataset shape: {df_featured.shape}")

    # Define target
    target_column = 'loan_status'

    # Separate features and target
    X = df_featured.drop(columns=[target_column])
    y = df_featured[target_column]

    # Define feature groups
    categorical_features = [
        'gender',
        'married',
        'dependents',
        'education',
        'self_employed',
        'property_area'
    ]

    numerical_features = [
        'applicant_income',
        'coapplicant_income',
        'loan_amount',
        'loan_term',
        'credit_history',
        'total_income',
        'income_loan_ratio',
        'monthly_loan_amount',
        'income_share_ratio',
        'high_income_flag'
    ]

    # Create preprocessor
    preprocessor = create_preprocessor(
        categorical_features=categorical_features,
        numerical_features=numerical_features
    )

    # Fit and transform
    X_transformed = preprocessor.fit_transform(X)
    logger.info("Successfully transformed feature set")

    # Ensure save directories exist
    output_path = Path(output_file)
    preprocessor_path = Path(preprocessor_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)

    # Save preprocessor
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Saved preprocessor to {preprocessor_path}")

    # Convert transformed data to DataFrame
    transformed_feature_names = preprocessor.get_feature_names_out()
    df_transformed = pd.DataFrame(
        X_transformed,
        columns=transformed_feature_names
    )

    # Add target column back
    df_transformed[target_column] = y.values

    # Save transformed data
    df_transformed.to_csv(output_path, index=False)
    logger.info(f"Saved transformed dataset to {output_path}")

    return df_transformed


if __name__ == "__main__":
    import argparse

    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description='Feature engineering for loan approval prediction dataset.'
    )

    parser.add_argument(
        '--input',
        default=project_root / 'data' / 'processed' / 'cleaned_loan_data.csv',
        help='Path to cleaned CSV file'
    )

    parser.add_argument(
        '--output',
        default=project_root / 'data' / 'processed' / 'featured.csv',
        help='Path for output CSV file (engineered features)'
    )

    parser.add_argument(
        '--preprocessor',
        default=project_root / 'models' / 'preprocessor.joblib',
        help='Path for saving the preprocessor object'
    )

    args = parser.parse_args()

    run_feature_engineering(
        input_file=args.input,
        output_file=args.output,
        preprocessor_file=args.preprocessor
    )
