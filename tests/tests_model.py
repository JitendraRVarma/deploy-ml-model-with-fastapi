"""
Test script for testing the model's functions: preprocess_data, train, and predict.
"""

import numpy as np
import pandas as pd
import pytest
from model.inference import predict
from model.train import preprocess_data, train
from pandas import testing as tm


@pytest.fixture
def test_df() -> pd.DataFrame:
    """
    Fixture function that provides a sample test DataFrame for testing.

    Returns:
        pd.DataFrame: A sample DataFrame for testing.
    """
    return pd.DataFrame(
        [
            {
                "age": 29,
                " workclass": " Private",
                " fnlgt": 128509,
                " education": " HS-grad",
                " education-num": 9,
                " marital-status": " Married-spouse-absent",
                " occupation": " Machine-op-inspct",
                " relationship": " Not-in-family",
                " race": " White",
                " sex": " Female",
                " capital-gain": 0,
                " capital-loss": 0,
                " hours-per-week": 38,
                " native-country": " El-Salvador",
                " salary": " <=50K",
            },
            {
                "age": 34,
                " workclass": " Federal-gov",
                " fnlgt": 190228,
                " education": " Bachelors",
                " education-num": 13,
                " marital-status": " Married-civ-spouse",
                " occupation": " Protective-serv",
                " relationship": " Husband",
                " race": " White",
                " sex": " Male",
                " capital-gain": 0,
                " capital-loss": 1902,
                " hours-per-week": 48,
                " native-country": " United-States",
                " salary": " >50K",
            },
            {
                "age": 48,
                " workclass": " Private",
                " fnlgt": 176732,
                " education": " 9th",
                " education-num": 5,
                " marital-status": " Divorced",
                " occupation": " Sales",
                " relationship": " Not-in-family",
                " race": " White",
                " sex": " Female",
                " capital-gain": 0,
                " capital-loss": 0,
                " hours-per-week": 40,
                " native-country": " United-States",
                " salary": " <=50K",
            },
            {
                "age": 47,
                " workclass": " Private",
                " fnlgt": 128796,
                " education": " HS-grad",
                " education-num": 9,
                " marital-status": " Married-civ-spouse",
                " occupation": " Craft-repair",
                " relationship": " Husband",
                " race": " White",
                " sex": " Male",
                " capital-gain": 0,
                " capital-loss": 0,
                " hours-per-week": 40,
                " native-country": " United-States",
                " salary": " >50K",
            },
            {
                "age": 53,
                " workclass": " Private",
                " fnlgt": 238481,
                " education": " Assoc-voc",
                " education-num": 11,
                " marital-status": " Married-civ-spouse",
                " occupation": " Exec-managerial",
                " relationship": " Husband",
                " race": " White",
                " sex": " Male",
                " capital-gain": 0,
                " capital-loss": 1485,
                " hours-per-week": 40,
                " native-country": " United-States",
                " salary": " <=50K",
            },
        ]
    )


def test_preprocess_data(test_df: pd.DataFrame):
    """
    Test function for preprocess_data.

    Args:
        test_df (pd.DataFrame): Sample test DataFrame.

    Returns:
        None
    """
    # Perform the preprocess_data function
    actual_df, _ = preprocess_data(test_df)

    # Expected salary series after preprocessing
    expected_salary = pd.Series([0, 1, 0, 1, 0], name="salary")

    # Assert that the processed salary series matches the expected one
    tm.assert_series_equal(actual_df["salary"], expected_salary)


def test_train(test_df: pd.DataFrame):
    """
    Test function for train.

    Args:
        test_df (pd.DataFrame): Sample test DataFrame.

    Returns:
        None
    """
    # Perform the train function
    actual_predict = train(test_df)

    # Expected predictions after training
    expected_predict = np.array([0, 1, 0, 1, 0])

    # Assert that the actual predictions match the expected predictions
    assert np.array_equal(actual_predict, expected_predict)


def test_predict(test_df: pd.DataFrame):
    """
    Test function for predict.

    Args:
        test_df (pd.DataFrame): Sample test DataFrame.

    Returns:
        None
    """
    # Perform the predict function
    actual_predict = predict(test_df.drop(" salary", axis=1))

    # Expected predictions after inference
    expected_predict = np.array([" <=50K", " >50K", " <=50K", " >50K", " <=50K"])

    # Assert that the actual predictions match the expected predictions
    assert np.array_equal(actual_predict, expected_predict)
