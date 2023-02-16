# bank_churn_project
Use machine learning to predict customer churn for a bank.

## Overview

We’ve been hired by a Norsk Bank to determine the likelihood of customer churn by using machine learning models. Our goal is to generate a model that has the best accuracy and f1 score which would provide the best method to determine the bank customers likelihood of remaining a bank customer or leaving the bank. Identifying a model that best predicts this can greatly improve customer retention which in turn will improve bank revenue. Banks generate revenue through selling products like bank loans and accounts to customers. Keeping and maintaining a good relationship with customers would ultimately mean customers will take on more bank products. 

Our team has found reliable data on Kaggle that will help us determine how to improve customer attrition. Using feature engineering and preprocessing to improve the readability of the data to bring relevant numbers to our model. The goal of our model is to find the best correlation with customer attrition. This data will be presented to the Norsk Bank to help them reduce customer churn. 


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

### Education

The education category 'Graduate' is vague/ambiguous 

In 2021, the highest level of education of the population age 25 and older in the United States was distributed as follows: 

  8.9% had less than a high school diploma or equivalent.
    
  our dataset shows 17% uneducated
        
  27.9% had high school graduate as their highest level of school completed. 
    
  our dataset shows 23.3%
        
  14.9% had completed some college but not a degree.
    
  ours only has an ambiguous 'graduate' as 36.6%. could be this, could mean post-grad, term is used interchangeably.
        
  10.5% had an associate degree as their highest level of school completed.
  
  our dataset has no distinction between associate/undergrad. just says 'college'. ours shows 11.9% for college 
        
  23.5% had a bachelors degree as their highest degree.
   
  ours shows 11.9% for college
        
  14.4% had completed an advanced degree such as a masters degree, professional degree or doctoral degree. 
    
  ours shows 6.1% post-graduate, 5.1% doctorate, and the ambiguous 36.6% 'graduate'
        
  It's hard to tell what they mean by 'graduate'. Unfortunately, its the bulk of our data, and dropping it will hurt our sample size. 

  The existence of a 'college' column suggests that graduate could mean beyond college. 
   
  Alternatively, college could mean you attended college but did not graduate, and 'graduate' represents standard undergrads.
        
   Its a high amount of the data (36.6%), so its likely a mix of multiple categories not included, i.e. trade schools, associate degrees, dropouts,

   specializations


### Feature Creation

Using feature creation to make a ratio between dependents and income sources

consider someone divorced as Single. only 10% of divorced people receive alimony payments

either a 1 (single/divorced) or a 2 (married)
divide that by the amount of dependents plus one, which represents how many people you take care of
        i.e. a single person with no dependents only takes care of themself, so they are a 1.
make a ratio between avg open to buy and credit limit
make a ratio between transaction amount and trans count
make a ratio between age and tenure
make a ratio between age and tenure
make a rank for the income as a 0-4 low-high
        
Scale all numerical/float values that don't represent categories

scaler = StandardScaler()

Use Column Transformer to scale only the numerical/float values that don't
represent categories (male/female, married/single/divorced)
Going to try and remove all things related to Edu/age/dependents, see what happens
col_tran= ColumnTransformer

Use SMOTE to add synthetic data and balance our target feature value count



## Model Evaluation

Best Model details

clf = GradientBoostingClassifier(
n_estimators=500,    #default = 100    range =
random_state=2,     #default = None   range =
subsample=1,     #default = 1   range =
min_samples_split = 4,      #default = 2   range = 2-inf
max_depth=3,  #default = 3    range = 1-inf
min_impurity_decrease=0,    #default = 0    range = 0 - inf
min_samples_leaf = 1,            #default = 1   range = 1 - inf
min_weight_fraction_leaf = 0,     #default = 0   range =0 - 0.5
max_leaf_nodes = None     #default = None   range = 2-inf
)


## Contributors

Austin Caras
Ben Harrington
Madhuri Krishna
Brian Peebles

