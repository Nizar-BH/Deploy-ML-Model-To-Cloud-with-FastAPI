# Script to train machine learning model.
import pickle
from sklearn.model_selection import train_test_split
from ml.data import process_data
import pandas as pd
import os
from ml.model import train_model, compute_model_metrics, inference, compute_slices

# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv("../data/census.csv")
data.columns = data.columns.str.replace(' ', '')
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder_test, lb_test = process_data(test, categorical_features=cat_features,
                                                     label="salary", training=False,
                                                     encoder=encoder, lb=lb)


save_path = "../model"

model = train_model(X_train, y_train)
pickle.dump(model, open(os.path.join(save_path, "trained_model.pkl"), "wb"))
pickle.dump(encoder, open(os.path.join(save_path, "encoder.pkl"), "wb"))
pickle.dump(lb, open(os.path.join(save_path, "labelizer.pkl"), "wb"))


# Evaluate the model and calculate the metrics for model performance.

preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test,preds)
print(f"Precision: {precision}, Recall: {recall}, Fbeta: {fbeta}")

# Compute the performance for slices.
for feature in cat_features:
    compute_slices(test, feature, y_test, preds)


