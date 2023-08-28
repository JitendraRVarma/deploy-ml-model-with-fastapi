import requests

# Define the API endpoint URL
api_url = "https://udacity-salary-predictor-fb9e03157e82.herokuapp.com/predict/"  # NOQA

# Sample input data for model inference
input_data = {
    "data": [
        {
            "age": 46,
            "workclass": "Private",
            "fnlgt": 369538,
            "education": "10th",
            "education_num": 6,
            "marital_status": "Married-civ-spouse",
            "occupation": "Transport-moving",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States",
        },
        {
            "age": 45,
            "workclass": "Private",
            "fnlgt": 172274,
            "education": "Doctorate",
            "education_num": 16,
            "marital_status": "Divorced",
            "occupation": "Prof-specialty",
            "relationship": "Unmarried",
            "race": "Black",
            "sex": "Female",
            "capital_gain": 0,
            "capital_loss": 3004,
            "hours_per_week": 35,
            "native_country": "United-States",
        },
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
        },
    ]
}

# Send a POST request to the API
response = requests.post(api_url, json=input_data)

# Get the result and status code from the response
result = response.json()
status_code = response.status_code

# Print the results
print("Model Inference Result:", result)
print("Status Code:", status_code)
