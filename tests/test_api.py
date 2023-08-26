from api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_get_root():
    response = client.get("/")
    assert response.status_code == 200


def test_predict_below_50k():
    data = {
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
            }
        ]
    }
    response = client.post("/predict/", json=data)
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert response.json()["predictions"][0] == " <=50K"


def test_error():
    data = {
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
            }
        ]
    }
    response = client.post("/predict/", json=data)
    print(response)
    assert response.status_code == 422
