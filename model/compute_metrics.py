import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

from model.inference import predict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the root directory of the project
ROOT_DIR = Path(__file__).resolve().parent.parent


def compute_metrics_on_slices(data: pd.DataFrame, feature_name: str) -> None:
    """
    Compute and write model metrics for slices of
    data based on a given categorical feature.

    Args:
        data (pd.DataFrame): Input data.
        feature_name (str): Name of the categorical feature.

    Returns:
        None
    """
    unique_values = data[feature_name].unique()
    file_path = Path(ROOT_DIR, "slice_output.txt")
    with open(file_path, "w") as f:
        for value in unique_values:
            logger.info(f"Computing Metrics on {feature_name}:{value}")
            subset = data[data[feature_name] == value]
            x_subset = subset.drop("salary", axis=1)
            y_subset = subset["salary"]

            predictions = predict(x_subset)
            report = classification_report(y_subset, predictions)

            f.write(f"Metrics for {feature_name} = {value}:\n")
            f.write(report)
            f.write("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    data_path = Path(ROOT_DIR, "data", "census.csv")
    data = pd.read_csv(data_path)
    data.columns = data.columns.str.strip()

    categorical_feature = "education"
    compute_metrics_on_slices(data, categorical_feature)
