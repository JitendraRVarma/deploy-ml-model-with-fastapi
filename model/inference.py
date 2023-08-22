import logging
from pathlib import Path

import joblib
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the root directory of the project
ROOT_DIR = Path(__file__).resolve().parent.parent


def predict(data):
    data.columns = data.columns.str.strip()
    assets_path = Path(ROOT_DIR, "asset", "saved_pipeline.pkl")
    loaded_assets = joblib.load(assets_path)

    # Access the loaded components
    target_encoder = loaded_assets["target_encoder"]
    model_pipeline = loaded_assets["model_pipeline"]

    salary = model_pipeline.predict(data)

    salary = target_encoder.inverse_transform(salary)

    return salary


if __name__ == "__main__":
    data_path = Path(ROOT_DIR, "data", "census.csv")
    data = pd.read_csv(data_path)
    data.columns = data.columns.str.strip()
    data.drop("salary", axis=1, inplace=True)
    print(predict(data))
