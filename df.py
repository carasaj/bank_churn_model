
import pandas as pd
import numpy as np
import hvplot.pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from imblearn.metrics import classification_report_imbalanced
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import svm

"""
***Two different dataframes start here***
"""
#FIRST DF - build DF with a function that uses oneHotEncoder to encode the categorical columns
# The Gender column and Attrition (target) column have been encoded to a 1 or 0

def buildDataFrame():
    
    churn = pd.read_csv('BankChurners.csv')
    churn['Attrition_Flag'] = churn['Attrition_Flag'].apply(encodeAttrition)
    churn['Gender'] = churn['Gender'].apply(encodeGender)
    
    
    #OneHotEncoder
    enc = OneHotEncoder(sparse=False)
    #Creating a list of the columns with categorical variables

    categorical_variables = ['Education_Level','Marital_Status','Income_Category','Card_Category']
    encoded_data = enc.fit_transform(churn[categorical_variables])
    encoded_df = pd.DataFrame(
    encoded_data,
    columns = enc.get_feature_names(categorical_variables)
    )
    #Scaling the numerical columns
    churn_scaled = StandardScaler().fit_transform(churn[['Total_Relationship_Count','Contacts_Count_12_mon','Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1',
                                                         'Total_Trans_Ct','Total_Ct_Chng_Q4_Q1']])
    #Create a dataframe with the scaled data
    churn_transformed = pd.DataFrame(churn_scaled, columns=['Total_Relationship_Count','Contacts_Count_12_mon','Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1',
                                                            'Total_Trans_Ct','Total_Ct_Chng_Q4_Q1'])
    
    
    churn_concat = pd.concat(
        [churn[['Attrition_Flag','Gender']],churn_transformed,
         encoded_df,
        ],
        axis=1
    )      
        #Create features list
    X = churn_concat.drop(columns=['Attrition_Flag'])

    #create target list
    y = churn_concat['Attrition_Flag']

    #set up train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y)    
   
    return X_train, X_test, y_train, y_test
     

def encodeAttrition(Attrition_Type):
    """
    This will encode the column so 0 is Existing Customer and 1 is Attrited Customer
    """
    if Attrition_Type == 'Existing Customer':
        return 0
    else: 
        return 1
    

#Encoding gender column
def encodeGender(gender):
    """
    encoding so 1 = M and 0 = F
    """
    if gender == 'M':
        return 1
    else:
        return 0

    
def buildDataFrame_2():
    
    
    churn = pd.read_csv('BankChurners.csv')
    churn['Attrition_Flag'] = churn['Attrition_Flag'].apply(encodeAttrition)
    churn['Gender'] = churn['Gender'].apply(encodeGender)
    
    card_dummies = pd.get_dummies(churn[['Education_Level','Marital_Status','Income_Category','Card_Category']])
    dummies_df = pd.DataFrame(card_dummies)
    
    #StandardScaler 
    churn_scaled = StandardScaler().fit_transform(churn[['Total_Relationship_Count','Contacts_Count_12_mon','Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1',
                                                         'Total_Trans_Ct','Total_Ct_Chng_Q4_Q1']])
    
    #Create a dataframe with the scaled data
    churn_transformed = pd.DataFrame(churn_scaled, columns=['Total_Relationship_Count','Contacts_Count_12_mon','Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1',
                                                            'Total_Trans_Ct','Total_Ct_Chng_Q4_Q1'])
    
    
    
    
    churn_dummies_concat = pd.concat(
        [churn[['Attrition_Flag','Gender']],churn_transformed,
         dummies_df
                 ],
        axis=1
)
       #Create features list
    X = churn_dummies_concat.drop(columns=['Attrition_Flag'])

    #create target list
    y = churn_dummies_concat['Attrition_Flag']

    #set up train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y)    
   
    return X_train, X_test, y_train, y_test
    

def encodeAttrition(Attrition_Type):
    """
    This will encode the column so 0 is Existing Customer and 1 is Attrited Customer
    """
    if Attrition_Type == 'Existing Customer':
        return 0
    else: 
        return 1
    

#Encoding gender column
def encodeGender(gender):
    """
    encoding so 1 = M and 0 = F
    """
    if gender == 'M':
        return 1
    else:
        return 0




    

    
    