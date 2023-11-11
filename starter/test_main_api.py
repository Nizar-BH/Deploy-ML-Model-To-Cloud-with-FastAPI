"""
Tests for the FastAPI application.

Uses pytest and FastAPI TestClient for testing the API endpoints.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))

from main import app
import logging
import pytest
from fastapi.testclient import TestClient


client = TestClient(app)


@pytest.fixture
def sample_data():
    """
    Fixture providing a sample data dictionary for testing for class 1.

    Returns:
    dict: A dictionary containing sample input data.
    """
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


@pytest.fixture
def sample_data2():
    """
    Fixture providing a sample data dictionary for testing for class 0.

    Returns:
    dict: A dictionary containing sample input data.
    """
    return {
        "age": 31,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "education_num": 14,
        "marital_status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 14084,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States"
    }


def test_welcome_message():
    """
    Test for the welcome message endpoint ("/").

    Checks if the response status code is 200 and if the returned message matches the expected message.
    """
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()[
        'message'] == "Hello! This is the last project of the Udacity MLops Nanodegree!"


def test_model_inference_class1(sample_data):
    """
    Test for model inference endpoint ("/predict") for class 1 prediction.

    Checks if the response status code is 200 and if the returned prediction matches the expected values.
    """
    r = client.post("/predict", json=sample_data)

    assert r.status_code == 200
    assert r.json()[0]["age"] == sample_data["age"]
    assert r.json()[0]["fnlgt"] == sample_data["fnlgt"]
    assert r.json()[0]["prediction"] == ' <=50K'


def test_model_inference_class_0(sample_data2):
    """
    Test for model inference endpoint ("/predict") for class 0 prediction.
    """

    r = client.post("/predict/", json=sample_data2)

    assert r.status_code == 200
    assert r.json()[0]["age"] == sample_data2["age"]
    assert r.json()[0]["fnlgt"] == sample_data2["fnlgt"]
    assert r.json()[0]["prediction"] == ' >50K'


def test_incomplete_inference_query():
    """
    Test for incomplete model inference query.

    Checks if the response status code is 422 (Unprocessable Entity) and if the 'prediction' key is not present in the response.
    """
    data = {
        "occupation": "Exec-managerial",
        "race": "Black",
        "capital_gain": 5178,
        "education": "9th"}
    r = client.post("/predict", json=data)
    assert r.status_code == 422
    assert 'prediction' not in r.json()["detail"][0].keys()

    logging.warning(
        f"The sample has {len(data)} features. Must be 14 features")
