
import argparse
import os
import pandas as pd
import joblib
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    xgboost_available = True
except ImportError:
    xgboost_available = False

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Train multiple classification models")

    parser.add_argument(
        '--data',
        type=str,
        default=project_root / 'data' / 'processed' / 'featured.csv',
        help='Path to processed CSV dataset'
    )

    parser.add_argument(
        '--models-dir',
        type=str,
        default=project_root / 'model',
        help='Directory to save best trained model'
    )

    return parser.parse_args()


def main(args):
    # Load dataset
    data = pd.read_csv(args.data)

    # Define target column
    target_column = 'loan_status'

    # Split features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Testing data shape: {X_test.shape}")

    # Models to compare
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(random_state=42)
    }

    if xgboost_available:
        models['xgboost'] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )

    best_model = None
    best_model_name = None
    best_accuracy = 0

    # Train and compare models
    for model_name, model in models.items():
        logger.info(f"Training {model_name}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"{model_name} accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = model_name

    logger.info(f"Best model: {best_model_name}")
    logger.info(f"Best accuracy: {best_accuracy:.4f}")

    # Save best model
    save_dir = os.path.join(args.models_dir, 'trained')
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, 'best_model.pkl')
    joblib.dump(best_model, save_path)

    logger.info(f"Saved best model to: {save_path}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
