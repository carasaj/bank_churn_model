# bank_churn_project

The main notebook is `bank_churn_final.ipynb`

The project goal is to create a supervised machine learning model for predicting bank customer churn. Alternative data processing techniques, features, and models are used as comparisons. 


## Overview

This readme will walk through how a binary classification model is used to predict a customers' likelihood of remaining a bank customer, also known as churn. Identifying and tuning the model to best predict churn can greatly improve customer retention, which in turn will improve bank revenue. Banks generate revenue through selling products like bank loans and accounts to customers. Keeping and maintaining a good relationship with customers would ultimately mean customers will take on more bank products. 

## Data Source

Bank data was sourced from Kaggle.

Main dataset:
https://www.kaggle.com/datasets/teralasowmya/bankchurner

Comparison dataset:
https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers
This dataset was used to explore how the techniques used in the original dataset might work on similar data. It has less features.

## Technology & Libraries Used

![Libraries](https://github.com/carasaj/bank_churn_project/blob/main/Resources/Libraries.PNG) 

https://scikit-learn.org/stable/supervised_learning.html#supervised-learning

https://imbalanced-learn.org/stable/install.html

https://xgboost.readthedocs.io/en/stable/install.html


## Data Preparation, Feature Engineering, and Model Training

### Feature Analysis

Each potential feature was run against the target in a 1:1 model to determine their correlation. This novel technique was used to try and discern which features were irrelevant.

Out of the original column data, the highest correlating feature was `Total_Revolving_Balance`.

While most variables were conserved as features in one way or another, this step proved useful for our initial analysis of the raw data.

### Data Cleanup

Initially, a handful of column names were changed for clarity and/or brevity. This was just to make the raw dataset easier to read.
MAYBE REMOVE THAT PART FROM CODE AND REMOVE THIS LINE. NO NEED FOR IT ANYMORE.

Any rows with `NaN` or `Unknown` values were dropped. 

### Education Category

This was the hardest category to rank. 

Ranking is more desirable than using a categorizer like `OneHotEncoder` because education is mostly cumulative. i.e. you need a high school degree before college, college before masters, etc.,

The `education` category `graduate` is ambiguous, so it's hard to tell what it means in the context of the other education categories.

The existence of a `college` value suggests that a `graduate` is beyond college. 
Alternatively, `college` could mean they attended college but did not graduate, and `graduate` actually represents standard undergrads.
The `post-graduate` and `doctorate` value also add to the confusion, making the diferences between a `graduate` and `post-graduate` unclear.
        
Its possibly a mix of multiple categories not included, i.e. trade schools, associate degrees, dropouts, or specializations. Unfortunately, its the bulk of our data, and dropping it will hurt our sample size. It was decided that the data would be kept, classifying `graduate` between `college` and `post-graduate`.

`Uneducated = 0`

`High School = 1`

`College = 2`

`Graduate = 3`

`Post-Graduate = 4`

`Doctorate = 5`


### Marital Dependent Ratio

First, we change marital_status values from `Single/Divorced/Married` to `1/1/2`, reflecting the amount of income sources in their household. The dependents  are kept as `int` values between `0 and 6`.

When both features are combined using the formula below, it results in a `float` between `0 and 1`. 

`( Marital_Status / (Dependents + 1) / 2 ) = marital_dependent_ratio`

This ratio reflects the amount of income vs dependents in a household. 

Essentially, a married couple with no dependents has the highest value of `1`. A single person with multiple dependents would have a much lower ratio.

### Credit Usage
make a ratio between avg open to buy and credit limit

### Average Transaction Value
make a ratio between transaction amount and trans count

### Tenure by Age
make a ratio between age and tenure

### Income Rank
make a rank for the income as a 0-4 low-high

### Standard Scaling with Column Transformer
Scale all numerical/float values that don't represent categories

Use Column Transformer to scale only the numerical/float values that don't
represent non-ranked categories (i.e. 0/1 for male/female are unranked categories)

### Oversampling
Use SMOTE to add synthetic data and balance our target feature value count



## Model Evaluation

Best Model details

clf = GradientBoostingClassifier(

n_estimators=500,                    #default = 100    range = 1-inf

random_state=2,                      #default = None   range = 1-inf

subsample=1,                         #default = 1   range = 0. - 1

min_samples_split = 4,               #default = 2   range = 2-inf

max_depth=3,                         #default = 3    range = 1-inf

min_impurity_decrease=0,             #default = 0    range = 0 - inf

min_samples_leaf = 1,                #default = 1   range = 1 - inf

min_weight_fraction_leaf = 0,        #default = 0   range =0 - 0.5
 
max_leaf_nodes = None                #default = None   range = 2-inf

)


## Contributors

Austin Caras
Ben Harrington
Madhuri Krishna
Brian Peebles

