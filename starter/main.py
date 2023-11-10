from typing import Literal
import pickle
import os
import sys

import uvicorn

sys.path.append(os.path.join(os.path.dirname(__file__), "starter", "ml"))
from fastapi import FastAPI
from pydantic import BaseModel
from data import process_data
import pandas as pd

SAVE_PATH = "./model"

# Initialize the app
app = FastAPI(title="Model API", description="This API takes in data and returns a prediction")


class ModelInput(BaseModel):
    age: int
    workclass: Literal['State-gov',
    'Self-emp-not-inc',
    'Private',
    'Federal-gov',
    'Local-gov',
    'Self-emp-inc',
    'Without-pay']
    fnlgt: int
    education: Literal[
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th',
        'Some-college',
        'Assoc-acdm', '7th-8th', 'Doctorate', 'Assoc-voc', 'Prof-school',
        '5th-6th', '10th', 'Preschool', '12th', '1st-4th']
    education_num: int
    marital_status: Literal["Never-married",
    "Married-civ-spouse",
    "Divorced",
    "Married-spouse-absent",
    "Separated",
    "Married-AF-spouse",
    "Widowed"]
    occupation: Literal["Tech-support",
    "Craft-repair",
    "Other-service",
    "Sales",
    "Exec-managerial",
    "Prof-specialty",
    "Handlers-cleaners",
    "Machine-op-inspct",
    "Adm-clerical",
    "Farming-fishing",
    "Transport-moving",
    "Priv-house-serv",
    "Protective-serv",
    "Armed-Forces"]
    relationship: Literal["Wife", "Own-child", "Husband",
    "Not-in-family", "Other-relative", "Unmarried"]
    race: Literal["White", "Asian-Pac-Islander",
    "Amer-Indian-Eskimo", "Other", "Black"]
    sex: Literal["Female", "Male"]
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: Literal[
        'United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
        'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran',
        'Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand',
        'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
        'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
        'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru',
        'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
        'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
        'Holand-Netherlands']

    class Config:
        json_schema_extra = {
            "example": {
                "age": 39,
                "workclass": 'State-gov',
                "fnlgt": 77516,
                "education": 'Bachelors',
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": " Adm-clerical",
                "relationship": " Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": 'United-States'
            }
        }


@app.on_event("startup")
async def startup_event():
    loaded_model, loaded_encoder, loaded_lb = load_saved_objects()

    if loaded_model is not None:
        global model, encoder, lb
        model, encoder, lb = loaded_model, loaded_encoder, loaded_lb


def load_saved_objects():
    model, encoder, lb = None, None, None
    try:
        with open(os.path.join(SAVE_PATH, "trained_model.pkl"), "rb") as model_file, \
                open(os.path.join(SAVE_PATH, "encoder.pkl"), "rb") as encoder_file, \
                open(os.path.join(SAVE_PATH, "labelizer.pkl"), "rb") as lb_file:
            model = pickle.load(model_file)
            encoder = pickle.load(encoder_file)
            lb = pickle.load(lb_file)
    except (FileNotFoundError, EOFError, pickle.PickleError) as e:
        print(f"Error loading saved objects: {e}")

    return model, encoder, lb


@app.get("/")
def read_root():
    return {"message": "Hello! This is the last project of the Udacity MLops Nanodegree!"}


@app.post("/predict")
def predict(input: ModelInput):
    Input_data = {'age': input.age,
                  'workclass': input.workclass,
                  'fnlgt': input.fnlgt,
                  'education': input.education,
                  'education-num': input.education_num,
                  'marital-status': input.marital_status,
                  'occupation': input.occupation,
                  'relationship': input.relationship,
                  'race': input.race,
                  'sex': input.sex,
                  'capital-gain': input.capital_gain,
                  'capital-loss': input.capital_loss,
                  'hours-per-week': input.hours_per_week,
                  'native-country': input.native_country,
                  }

    # Convert input data into dataframe
    input_df = pd.DataFrame([Input_data])

    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    # Process input data
    X, _, _, _ = process_data(input_df,
                              categorical_features=categorical_features,
                              training=False,
                              encoder=encoder,
                              lb=lb
                              )
    # Make the prediction and process it into a human readable format
    pred = model.predict(X)
    pred = lb.inverse_transform(pred)[0]

    return {"The income prediction is": pred}

if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)