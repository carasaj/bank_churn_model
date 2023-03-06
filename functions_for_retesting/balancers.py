import pandas as pd
import numpy as np
import hvplot.pandas
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from imblearn.metrics import classification_report_imbalanced
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

"""
LogisticRegression Classifier models start here. The first
oversamples the data, the second undersamples.
"""


def sample_1(RandomOverSampler, LogisticRegression, X, y, X_test, y_test):
    random_oversampler = RandomOverSampler(random_state=1)
    X_resampled, y_resampled = random_oversampler.fit_resample(X, y)

    #Fit the model
    lr_model = LogisticRegression()
    lr_model.fit(X_resampled, y_resampled)
    #Let the model make its predictions on the features
    predictions = lr_model.predict(X_resampled)

    
    #predcit testing features
    y_pred = lr_model.predict(X_test)

    
    print(classification_report_imbalanced(y_test, y_pred))




#LR with undersampled data

def sample_2(RandomUnderSampler, LogisticRegression, X_train, y_train, X_test, y_test):
    #undersample the data
    random_undersampler = RandomUnderSampler(random_state=1)
    X_resampled, y_resampled = random_undersampler.fit_resample(X_train, y_train)

    #Fit the model
    lr_model = LogisticRegression()
    lr_model.fit(X_resampled, y_resampled)
    #let the model make its predictions on the features
    predictions = lr_model.predict(X_resampled)

    
    #predcit testing features
    y_pred = lr_model.predict(X_test)

    
    