import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data-processor')


def load_data(file_path):
    """Load data from a CSV file."""
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)


def clean_data(df):
    """Clean the dataset by handling missing values, duplicates, and outliers."""
    logger.info("Cleaning dataset")

    # Make a copy to avoid modifying the original dataframe
    df_cleaned = df.copy()

    # Standardize column names
    df_cleaned.columns = [col.strip().lower() for col in df_cleaned.columns]

    # Remove duplicate rows
    duplicate_count = df_cleaned.duplicated().sum()
    if duplicate_count > 0:
        logger.info(f"Found {duplicate_count} duplicate rows")
        df_cleaned = df_cleaned.drop_duplicates()
        logger.info(f"Removed duplicates. New shape: {df_cleaned.shape}")

    # Handle missing values
    for column in df_cleaned.columns:
        missing_count = df_cleaned[column].isnull().sum()

        if missing_count > 0:
            logger.info(f"Found {missing_count} missing values in {column}")

            # Numeric columns → fill with median
            if pd.api.types.is_numeric_dtype(df_cleaned[column]):
                median_value = df_cleaned[column].median()
                df_cleaned[column] = df_cleaned[column].fillna(median_value)
                logger.info(
                    f"Filled missing values in {column} with median: {median_value}"
                )

            # Categorical columns → fill with mode
            else:
                mode_value = df_cleaned[column].mode()[0]
                df_cleaned[column] = df_cleaned[column].fillna(mode_value)
                logger.info(
                    f"Filled missing values in {column} with mode: {mode_value}"
                )

    # Handle outliers for important numerical features
    numerical_columns = [
        'applicant_income',
        'coapplicant_income',
        'loan_amount',
        'loan_term'
    ]

    for column in numerical_columns:
        if column in df_cleaned.columns:
            Q1 = df_cleaned[column].quantile(0.25)
            Q3 = df_cleaned[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_count = df_cleaned[
                (df_cleaned[column] < lower_bound) |
                (df_cleaned[column] > upper_bound)
            ].shape[0]

            if outlier_count > 0:
                logger.info(f"Found {outlier_count} outliers in {column}")

                # Cap outliers instead of removing rows
                df_cleaned[column] = np.where(
                    df_cleaned[column] < lower_bound,
                    lower_bound,
                    np.where(
                        df_cleaned[column] > upper_bound,
                        upper_bound,
                        df_cleaned[column]
                    )
                )

                logger.info(f"Capped outliers in {column}")

    # Convert target column into binary format if needed
    if 'loan_status' in df_cleaned.columns:
        df_cleaned['loan_status'] = df_cleaned['loan_status'].map({
            'Y': 1,
            'N': 0
        })
        logger.info("Converted loan_status to binary format")

    # Convert credit history to integer if possible
    if 'credit_history' in df_cleaned.columns:
        df_cleaned['credit_history'] = df_cleaned['credit_history'].astype(int)

    logger.info(f"Final cleaned dataset shape: {df_cleaned.shape}")

    return df_cleaned


def process_data(input_file, output_file):
    """Full data processing pipeline."""

    # Create output directory if it doesn't exist
    output_path = Path(output_file).parent
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(input_file)
    logger.info(f"Loaded data with shape: {df.shape}")

    # Clean data
    df_cleaned = clean_data(df)

    # Save processed data
    df_cleaned.to_csv(output_file, index=False)
    logger.info(f"Saved processed data to {output_file}")

    return df_cleaned


if __name__ == "__main__":
    process_data(
        input_file="../data/raw/loan_data.csv",
        output_file="../data/processed/cleaned_loan_data.csv"
    )
