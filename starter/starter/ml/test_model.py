import pytest, os, logging, pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
# from model import compute_model_metrics, inference

@pytest.fixture(scope="module")

def data():
    # code to load in the data.
    datapath = "../../data/census.csv"
    data = pd.read_csv(datapath)
    data.columns = data.columns.str.replace(' ', '')
    return data



@pytest.fixture(scope="module")
def path():
    return "../../data/census.csv"

@pytest.fixture(scope="module")
def model():
    # load model
    modelpath = "../../model/trained_model.pkl"
    return pickle.load(open(modelpath, "rb"))

@pytest.fixture(scope="module")
def features():

    cat_features = [    "workclass",
                        "education",
                        "marital-status",
                        "occupation",
                        "relationship",
                        "race",
                        "sex",
                        "native-country"]
    return cat_features

@pytest.fixture(scope="module")
def train_data(data, features):

    train, test = train_test_split( data,
                                test_size=0.20,
                                random_state=0,
                                )
    X_train, y_train, encoder, lb = process_data(
                                            train,
                                            categorical_features=features,
                                            label="salary",
                                            training=True
                                        )
    return X_train, y_train
def test_import_data(path):
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        logging.error("Dataset not found, check your path")
        raise FileNotFoundError
    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
    except AssertionError:
        logging.error("Dataset is empty")
        raise AssertionError


def test_features(data,features):
    """"Test that the categorical features exist in the dataset"""
    try:
        assert all(feature in data.columns for feature in features)
    except AssertionError:
        logging.error("Features are not correctly identified")
        raise AssertionError

def test_model_can_fit(model,train_data):
    """Test that the model is fitted"""
    X_train, _ = train_data
    try:
        assert model.predict(X_train)
    except NotFittedError:
        logging.error("Model is not fitted!")
        raise NotFittedError

def test_inference(model,X_test):
    """Test that the model can make predictions"""
    try:
        preds = inference(model, X_test)
        assert isinstance(preds, list)
    except AssertionError:
        logging.error("Model can't make predictions")
        raise AssertionError

def test_compute_model_metrics(X_test, y_test):
    """Test that the model can compute metrics"""
    try:
        preds = inference(model, X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, preds)
        assert isinstance(precision, float)
        assert isinstance(recall, float)
        assert isinstance(fbeta, float)
    except AssertionError:
        logging.error("Model can't compute metrics")
        raise AssertionError

# def test_compute_performance_for_slices(data,features):
#     """Test that the model can compute metrics"""
#     try:
#         for feature in features:
#             compute_slices(data, feature, y_test, preds)
#     except AssertionError:
#         logging.error("Model can't compute slices")
#         raise AssertionError