from main import app
import logging
import pytest
from fastapi.testclient import TestClient
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
client = TestClient(app)


@pytest.fixture
def sample_data():
    return {
        "age": 50,
        "workclass": "Private",
        "fnlgt": 234721,
        "education": "Doctorate",
        "education_num": 16,
        "marital_status": "Separated",
        "occupation": "Exec-managerial",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States"
    }


def test_welcome_message():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()[
        'message'] == "Hello! This is the last project of the Udacity MLops Nanodegree!"


def test_model_inference(sample_data):

    r = client.post("/predict", json=sample_data)

    assert r.status_code == 200
    assert r.json()[0]["age"] == sample_data["age"]
    assert r.json()[0]["fnlgt"] == sample_data["fnlgt"]
    assert r.json()[0]["prediction"] == ' <=50K'


def test_model_inference_class_0(sample_data):
    sample_data["education"] = "HS-grad"
    sample_data["education_num"] = 1
    sample_data["occupation"] = "Handlers-cleaners"
    sample_data["sex"] = "Male"
    sample_data["hours_per_week"] = 35

    r = client.post("/predict/", json=sample_data)

    assert r.status_code == 200
    assert r.json()[0]["age"] == sample_data["age"]
    assert r.json()[0]["fnlgt"] == sample_data["fnlgt"]
    assert r.json()[0]["prediction"] == ' <=50K'


def test_incomplete_inference_query():
    data = {
        "occupation": "Exec-managerial",
        "race": "Black",
        "capital_gain": 5178,
        "education": "9th"}
    r = client.post("/predict", json=data)
    # Check that the response status code is 422 (Unprocessable Entity)
    assert r.status_code == 422
    assert 'prediction' not in r.json()["detail"][0].keys()

    logging.warning(
        f"The sample has {len(data)} features. Must be 14 features")
