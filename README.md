# bank_churn_project

The main file is bank_churn_final.ipynb

Our project is to create a machine learning model for predicting customer churn for a bank. We will experiment with various processing techniques and models to maximize accuracy. Our best model will be compared to other attempts using similar models to show how each step impacts the accuracy. 



## Overview

Our goal is to generate a model that has the best f1 score , which is used to determine the customers' likelihood of remaining a bank customer or leaving the bank. Identifying and tuning a model to best predict this can greatly improve customer retention, which in turn will improve bank revenue. Banks generate revenue through selling products like bank loans and accounts to customers. Keeping and maintaining a good relationship with customers would ultimately mean customers will take on more bank products. 

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

The idea behind looking at a single variable from the many was to help us determine if there are variables that are clearly usesless

provide little to no utlility. Going through all the columns individually along with attrition/Status of the cusrtomer provided excelent data. 

Train_test_split was used along with SVC, LogisticRegression and RandomForestClassifier models were used to find hte best accuracy, precision, 

recall and f1 scores results. The one variable that outshined all others was Total_Revolving_Balance with the highest overall combination of results.

As seen in the final code, this variable is in use and the least useful variables have been dropped to help improve our final results.


### Education

The education category 'Graduate' is vague/ambiguous 

In 2021, the highest level of education of the population age 25 and older in the United States was distributed as follows:



  8.9% had less than a high school diploma or equivalent.
  
  
  Our dataset shows 17% uneducated
  
  
  27.9% had high school graduate as their highest level of school completed
  
  
  Our dataset shows 23.3%        
  
  
  14.9% had completed some college but not a degree.
  
  
  Ours only has an ambiguous 'graduate' as 36.6%. could be this, could mean post-grad, term is used interchangeably.    
  
  
  10.5% had an associate degree as their highest level of school completed.  
  
  
  Our dataset has no distinction between associate/undergrad. just says  11.9% for college     
  
  
  23.5% had a bachelors degree as their highest degree.   
  
  
  Ours shows 11.9% for college        
  
  
  14.4% had completed an advanced degree such as a masters degree, professional degree or doctoral degree.     
  
  
  Ours shows 6.1% post-graduate, 5.1% doctorate, and the ambiguous 36.6% 'graduate'        
  
  It's hard to tell what they mean by 'graduate'. Unfortunately, its the bulk of our data, and dropping it will hurt our sample size. 
  The existence of a 'college' column suggests that graduate could mean beyond college. 
   
  Alternatively, college could mean you attended college but did not graduate, and 'graduate' represents standard undergrads.
        
  Its a high amount of the data (36.6%), so its likely a mix of multiple categories not included, i.e. trade schools, associate degrees, dropouts, or specializations.


### Feature Engineering

Feature creation to make a ratios, rankings, or 0/1 boolean classifier. Original features are dropped in favor of the newly created features.

Consider someone divorced as Single. Either a 1 (single/divorced) or a 2 (married)
  Divide that by the amount of dependents plus one, then divide that by two.

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

