"""
FastAPI Application for Salary Prediction

This module defines a FastAPI application
that provides an API for salary prediction
using a pre-trained model.

Usage:
1. Start the FastAPI server using `uvicorn`.
2. Send POST requests to the '/predict/' endpoint with
input data to get salary predictions.

For more information about the API, visit http://localhost:8000/docs.
"""

from typing import List

import pandas as pd
import uvicorn
from fastapi import FastAPI
from model.inference import predict
from pydantic import BaseModel

app = FastAPI()


class InputItem(BaseModel):
    """Input data item for prediction"""

    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 20,
                    "workclass": "Private",
                    "fnlgt": 44064,
                    "education": "Some-college",
                    "education_num": 10,
                    "marital_status": "Never-married",
                    "occupation": "Prof-specialty",
                    "relationship": "Own-child",
                    "race": "White",
                    "sex": "Male",
                    "capital_gain": 0,
                    "capital_loss": 0,
                    "hours_per_week": 25,
                    "native_country": "United-States",
                }
            ]
        }
    }


@app.put("/items/{item_id}")
async def update_item(item_id: int, item: InputItem):
    results = {"item_id": item_id, "item": item}
    return results


class Item(BaseModel):
    """List of input items"""

    data: List[InputItem]


@app.get("/")
async def read_root():
    """Root endpoint for API greeting"""
    return {"message": "Welcome to the Salary Prediction API"}


@app.post("/predict/")
async def predict_data(received_data: Item):
    """
    Predict the salary based on the input data.

    Args:
        received_data (Item): List of input items.

    Returns:
        dict: Predicted salaries.
    """
    input_item = received_data.dict()
    df = pd.DataFrame(input_item["data"])
    df.columns = df.columns.str.replace("_", "-")
    predictions = predict(df)
    return {"predictions": list(predictions)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
