# bank_churn_model

The main notebook is `final_model.ipynb`

The goal is to create a supervised machine learning model for predicting bank customer churn. Alternative data processing techniques, features, and models are used as comparisons. 


## Overview

This readme will walk through how a binary classification model is used to predict if a customer will end their services at bank, also known as churn. Identifying and tuning the model to best predict churn can greatly improve customer retention, which in turn will improve bank revenue.

## Data Source

Bank data was sourced from Kaggle.

[Main dataset](https://www.kaggle.com/datasets/teralasowmya/bankchurner)

[Comparison dataset](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers)

This second dataset contains similar data on bank customer churn. It has less features, and was only used as a comparison and reference.

## Technology & Libraries Used

[pandas](https://pandas.pydata.org/docs/)

[pathlib](https://docs.python.org/3/library/pathlib.html)

[scikit-learn](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)

[imblearn](https://imbalanced-learn.org/stable/install.html)

## Data Preparation, Feature Engineering, and Model Training

### Data Cleanup

Rows with excessive Unknown values were dropped to preserve the sample size. No further data cleaning was necessary.

### Feature Analysis

These tests can be found in the feature_selection folder.

Features were evaluated using the following techniques:

Each potential feature was run against the target in a 1:1 model to determine their correlation and  discern which features were irrelevant.

Sequential Feature Selection was run 'forwards' to determine the most optimal features.

Fisher scores were generated.

A heatmap was generated to show correlation between features/features and features/target.

### Data Processing
Scale numerical/float values with StandardScaler
SMOTE to add synthetic data and balance our target feature value count
OneHotEncoder to encode categorical features

## Hyperparameters

GridSearch was done to explore the best possible values. The following were changed from default values:

`n_estimators=500`

`random_state= 2`

## Contributors

Austin Caras

Ben Harrington

Madhuri Krishna

Brian Peebles

