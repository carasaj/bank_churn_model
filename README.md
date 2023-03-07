# bank_churn_project

The main notebook is `bank_churn_final.ipynb`

The project goal is to create a supervised machine learning model for predicting bank customer churn. Alternative data processing techniques, features, and models are used as comparisons. 


## Overview

This readme will walk through how a binary classification model is used to predict a customers' likelihood of remaining a bank customer, also known as churn. Identifying and tuning the model to best predict churn can greatly improve customer retention, which in turn will improve bank revenue.

## Data Source

Bank data was sourced from Kaggle.

[Main dataset](https://www.kaggle.com/datasets/teralasowmya/bankchurner)

[Comparison dataset](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers)

This second dataset was used to explore how the techniques used in the original dataset might work on similar data. It has less features.

## Technology & Libraries Used

![Libraries](https://github.com/carasaj/bank_churn_project/blob/main/Resources/Libraries.PNG) 

[scikit-learn](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)

[imblearn](https://imbalanced-learn.org/stable/install.html)

[xgboost](https://xgboost.readthedocs.io/en/stable/install.html)


## Data Preparation, Feature Engineering, and Model Training

### Data Cleanup

Rows with excessive Unknown values were dropped to preserve the sample size. No further data cleaning was necessary.

### Feature Analysis

Each potential feature was run against the target in a 1:1 model to determine their correlation. This novel technique was used to try and discern which features were irrelevant.

Out of the original column data, the highest correlating feature was `Total_Revolving_Balance`.

While most variables were conserved as features in one way or another, this step proved useful for our initial analysis of the raw data.


### Standard Scaling with Column Transformer
Scale numerical/float values

Use Column Transformer to target specific features for scaling

### Oversampling
Use SMOTE to add synthetic data and balance our target feature value count


## Hyperparameters

No major hyperparameter tuning was needed. While GridSearch was done to explore the best possible values, only the following were changed:

`n_estimators=500`

`random_state= 2`

## Model Evaluation

## Contributors

Austin Caras

Ben Harrington

Madhuri Krishna

Brian Peebles

