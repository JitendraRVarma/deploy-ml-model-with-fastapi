import logging
from typing import List, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import logging
from typing import List, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer to perform label encoding on specified columns."""

    def __init__(self, columns: List[str]):
        self.columns = columns
        self.label_encoders = {}

    def fit(
        self, features: pd.DataFrame, label: Union[pd.Series, None] = None
    ) -> "LabelEncoderTransformer":
        """Fit label encoders on the specified columns."""
        for col in self.columns:
            logger.info(f"Fit Label encoding {col}")
            le = LabelEncoder()
            le.fit(features[col])
            self.label_encoders[col] = le
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform the features using label encoding."""
        x_encoded = features.copy()
        for col, le in self.label_encoders.items():
            logger.info(f"Transforimg Label encoding {col}")
            x_encoded[col] = x_encoded[col].apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            x_encoded[col] = le.transform(x_encoded[col])
        return x_encoded


class CustomDataPipeline(BaseEstimator, TransformerMixin):
    """Custom pipeline for label encoding and Random Forest classification."""

    def __init__(self, categorical_columns: List[str], target_column: str):
        self.categorical_columns = categorical_columns
        self.target_column = target_column
        self.pipeline = None
        self.label_encoder = LabelEncoderTransformer(self.categorical_columns)

    def fit(self, features: pd.DataFrame, label: pd.Series) -> "CustomDataPipeline":
        """Fit the label encoder and Random Forest classifier."""
        self.label_encoder.fit(features)
        logger.info(f"Running Pipeline")
        self.pipeline = Pipeline(
            [
                ("label_encoder", self.label_encoder),
                ("random_forest", RandomForestClassifier(random_state=42)),
            ]
        )
        self.pipeline.fit(features, label)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform the features using the label encoder."""
        return self.label_encoder.transform(features)

    def predict(self, features: pd.DataFrame) -> Union[pd.Series, None]:
        """Predict the target using the trained pipeline."""
        return self.pipeline.predict(features)
