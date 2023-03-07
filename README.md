# bank_churn_project

The main file is bank_churn_final.ipynb

Our project is to create a machine learning model for predicting bank customer churn. We will experiment with various processing techniques and models to maximize accuracy, precision, and recall. Our best model will be compared to other attempts using similar models to show how each step impacts the accuracy. 



## Overview

Our goal is to generate a model to predict a customers' likelihood of remaining a bank customer. Identifying and tuning a model to best predict this can greatly improve customer retention, which in turn will improve bank revenue. Banks generate revenue through selling products like bank loans and accounts to customers. Keeping and maintaining a good relationship with customers would ultimately mean customers will take on more bank products. 

Our team has found reliable data on Kaggle that will help us determine how to improve customer attrition. We will use feature engineering and preprocessing to improve the readability of the data and bring relevant numbers to our model. The goal of our model is to find the best correlation with customer attrition. This data will be presented to the Norsk Bank to help them reduce customer churn. 


## Data Source
Kaggle page
https://www.kaggle.com/datasets/teralasowmya/bankchurner

Comparison Dataset
https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers

## Technology & Libraries Used

![Libraries](https://github.com/carasaj/bank_churn_project/blob/main/Resources/Libraries.PNG) 

https://scikit-learn.org/stable/supervised_learning.html#supervised-learning

https://imbalanced-learn.org/stable/install.html

https://xgboost.readthedocs.io/en/stable/install.html


## Data Preparation and Model Training

### Comparing Each Variable to Attrition

Each potential feature was run against the target in a 1:1 model to determine their correlation. This novel technique was used to try and discern which features would piositively impact the model's predictions.

Train_test_split was used along with SVC, LogisticRegression and RandomForestClassifier models were used to find the best accuracy, precision, recall and f1 scores results. The one variable that outshined all others was Total_Revolving_Balance with the highest overall combination of results.

While most variables were conserved as features in one way or another, this step proved useful for our initial analysis of the raw data.

### Education

The education category 'Graduate' is ambiguous. It's hard to tell what they mean by 'graduate' in the context of the other education categories.

The existence of a 'college' column suggests that a 'graduate' is beyond college. 
Alternatively, 'college' could mean they attended college but did not graduate, and 'graduate' actually represents standard undergrads.
        
Its possibly a mix of multiple categories not included, i.e. trade schools, associate degrees, dropouts, or specializations. Unfortunately, its the bulk of our data, and dropping it will hurt our sample size. It was decided that the data would be kept, classifying 'graduates' between 'college' and 'post-graduate'.

Uneducated = 0
High School = 1
College = 2
Graduate = 3
Post-Graduate = 4
Doctorate = 5



### Feature Engineering

Feature creation to make a ratios, rankings, or 0/1 boolean classifier. Original features are dropped in favor of the newly created features.

( Marital_Status / (Dependents + 1) / 2 ) = marital_dependent_ratio
First, we change marital_status values from 'Single/Divorced/Married' to '1/1/2', reflecting the amount of income sources in their household.
When both features are combined using the formula above, it results in a value between 0 and 1. This ratio reflects the amount of income vs dependents in a household. Essentially, a married couple with no dependents has the highest value of 1. A single person with multiple dependents would have a much lower ratio.


make a ratio between avg open to buy and credit limit

make a ratio between transaction amount and trans count

make a ratio between age and tenure

make a rank for the income as a 0-4 low-high
        
Scale all numerical/float values that don't represent categories

Use Column Transformer to scale only the numerical/float values that don't
represent categories (male/female)

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

