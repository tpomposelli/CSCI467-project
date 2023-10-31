import pandas as pd
import numpy as np
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import argparse
import sys
from numpy.linalg import pinv
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("final_data.csv")


def main():

    #Set conditions to classify every player season
    conditions = [
        (df['Salary S'] <= 4),
        (df['Salary S'] <= 9) & (df['Salary S'] > 4),
        (df['Salary S'] <= 14) & (df['Salary S'] > 9),
        (df['Salary S'] <= 20) & (df['Salary S'] > 14),
        (df['Salary S'] <= 26) & (df['Salary S'] > 20),
        (df['Salary S'] <= 32) & (df['Salary S'] > 26),
        (df['Salary S'] <= 40) & (df['Salary S'] > 32),
        (df['Salary S'] > 40)
    ]
 
    outcomes = [0, 1, 2, 3, 4, 5, 6, 7]

    df['contract'] = np.select(conditions, outcomes, default=0)


    X = df.drop(columns={'Player','Tm','Salary S','Season S-1','% of Cap S', 'Salary S-1', 'Cap Maximum S', 'Cap Maximum S-1','% of Cap S-1','% of Cap S', 'contract'})
    y = df[['contract']]

    X_train, X_t, y_train, y_t = train_test_split(X, y, test_size=0.3, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(X_t, y_t, test_size=0.33, random_state=42)

    print(y.value_counts())
    print(y_dev.value_counts())
    print(y_test.value_counts())
    print(y_train.value_counts())
    print(y.count())
    print(y_dev.count())
    print(y_test.count())
    print(y_train.count())
    

    



main()