# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The used model for preidiction is a RandomForest classifier from the sklearn library. The defualt hyperparmeters are in use.
## Intended Use
This model demonstrates the capability to forecast an individual's salary level by leveraging different features.
## Training Data
The training data is from the UCI Machine Learning Repository. The data is from the 1994 Census database. The data was collected by Barry Becker from the 1994 Census database. The data set contains 48,842 instances and 14 attributes. It is available at https://archive.ics.uci.edu/ml/datasets/census+income.
## Evaluation Data
For the evaulation data sliced the main dataset into train and test, with a 80/20 split. The test data is used to evaluate the model's performance.
As for the processing of the data, we put in place categorical encoding using onehot encoding and binazier was used for the target variable.
## Metrics
The model achieved the following scores:
```
Precision: 0.72
Recall: 0.62
Fbeta: 0.665
```

## Ethical Considerations
Model performance should be treated and analyzed with caution as the training features included 
infos such as race and sex. This could lead to biased predictions.
## Caveats and Recommendations
The only use case for this model is to predict the salary level of an individual. The model should not be used for any other purpose.