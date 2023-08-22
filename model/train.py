import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from model.pipeline import CustomDataPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the root directory of the project
ROOT_DIR = Path(__file__).resolve().parent.parent


def preprocess_data(data: pd.DataFrame) -> tuple:
    """
    Preprocess the data by splitting it into
    features and labels, and performing label encoding.

    Args:
        data (pd.DataFrame): The input data.

    Returns:
        clean_data (pd.DataFrame): The preprocessed data.
        Label encoder
    """
    logger.info("Preprocessing data")

    # Create a copy of the original data
    clean_data = data.copy()

    # Strip whitespaces in column names
    clean_data.columns = clean_data.columns.str.strip()

    # Perform label encoding for 'salary' column
    salary_encoder = LabelEncoder()
    clean_data["salary"] = salary_encoder.fit_transform(clean_data["salary"])

    logger.info("Data preprocessing complete")
    return clean_data, salary_encoder


def train(data: pd.DataFrame = None):
    """Load, preprocess, and train the model."""
    if data is None:
        data_path = Path(ROOT_DIR, "data", "census.csv")
        data = pd.read_csv(data_path)

    clean_data, salary_encoder = preprocess_data(data)
    target_column = "salary"

    x_data = clean_data.drop(target_column, axis=1).copy()
    y_data = clean_data[target_column].copy()
    categorical_columns = x_data.columns[x_data.dtypes == object].tolist()

    # Define AUC scoring
    auc_scorer = make_scorer(roc_auc_score)

    # Initialize the custom pipeline
    pipeline = CustomDataPipeline(categorical_columns, target_column)

    # Perform cross-validation
    cv_scores = cross_val_score(pipeline, x_data, y_data, 3, auc_scorer)
    logger.info("Cross-validation scores: %s", cv_scores)
    logger.info("Saving assets")

    # fit model on entire data
    pipeline.fit(x_data, y_data)
    predict = pipeline.predict(x_data)

    # Save encoder and pipeline using joblib
    assets_path = Path(ROOT_DIR, "asset", "saved_pipeline.pkl")
    joblib.dump(
        {"target_encoder": salary_encoder, "model_pipeline": pipeline},
        assets_path,
    )

    logger.info("Assets saved successfully")

    return predict


if __name__ == "__main__":
    train()
