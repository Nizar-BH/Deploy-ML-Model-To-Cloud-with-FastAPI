import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "starter", "ml"))

import joblib
import pytest, logging
import pandas as pd

from data import process_data
from model import train_model, compute_model_metrics, inference, compute_slices
from sklearn.model_selection import train_test_split

DATA_PATH = "data/census.csv"
MODEL_PATH = "model/trained_model.pkl"


@pytest.fixture()
def data():
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.replace(' ', '')
    return data


@pytest.fixture()
def model():
    # load model
    return joblib.load(MODEL_PATH)


@pytest.fixture()
def features():
    cat_features = ["workclass",
                    "education",
                    "marital-status",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "native-country"]
    return cat_features


@pytest.fixture()
def data_train_test(data, features):
    train, test = train_test_split(data,
                                       test_size=0.20,
                                       random_state=0,
                                       )
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=features, label="salary", training=True
    )

    X_test, y_test, encoder, lb = process_data(
        test,
        categorical_features=features,
        label="salary",
        training=True,
        encoder=encoder,
        lb=lb,
    )
    return X_train, X_test, y_train, y_test



def test_import_data():
    try:
        data = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        logging.error("Dataset not found, check your path")
        raise FileNotFoundError
    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
    except AssertionError:
        logging.error("Dataset is empty")
        raise AssertionError


def test_features(data, features):
    """"Test that the categorical features exist in the dataset"""
    try:
        assert all(feature in data.columns for feature in features)
    except AssertionError:
        logging.error("Features are not correctly identified")
        raise AssertionError


def test_model_can_predict(model, data_train_test):
    """Test that the model can predict"""
    _, X_test, _, _ = data_train_test
    try:
        assert model.predict(X_test)
    except:
        logging.error("Model is not fitted!")


def test_compute_model_metrics(model, data_train_test):
    """Test that the model can compute metrics"""
    try:
        _, X_test, _, y_test = data_train_test
        preds = inference(model, X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, preds)
        assert isinstance(precision, float)
        assert isinstance(recall, float)
        assert isinstance(fbeta, float)
    except AssertionError:
        logging.error("Model can't compute metrics")
        raise AssertionError


def test_compute_performance_for_slices(data,data_train_test,features, model):
    """Test that the model can compute metrics"""
    try:
        _, test = train_test_split(data, test_size=0.20)
        _, X_test, _, y_test = data_train_test
        preds = inference(model, X_test)
        for feature in features:
            compute_slices(test, feature, y_test, preds)
        # Check that the file is created
        assert os.path.exists('./slice_output.txt')
    except AssertionError:
        logging.error("Model can't compute slices")
        raise AssertionError
