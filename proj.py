import pandas as pd
import numpy as np
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import argparse
import sys
from numpy.linalg import pinv
from sklearn.model_selection import GridSearchCV


df = pd.read_csv("final_data.csv")


def main():
    X = df.drop(columns={'Player','Tm','Salary S','Season S-1','% of Cap S', 'Salary S-1', 'Cap Maximum S', 'Cap Maximum S-1','% of Cap S-1','% of Cap S'})
    y = df[['% of Cap S']]
    X_train, X_t, y_train, y_t = train_test_split(X, y, test_size=0.3, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(X_t, y_t, test_size=0.33, random_state=42)

    for col in X:
        print(col)
        model = LinearRegression()
        model.fit(X_train[[col]], y_train)
        r_sq = model.score(X_dev[[col]], y_dev)
        y_preds = model.predict(X_dev[[col]])
        rmse = mean_squared_error(y_dev, y_preds, squared=False)
        print("The accuracy for", col, "is", r_sq, "RMSE:", rmse)

    maxValues = df[['% of Cap S', 'Salary S']].max()
    
    # Separating training and testing bases
    model = LinearRegression()
    model.fit(X_train, y_train)
    r_sq = model.score(X_dev, y_dev)
    y_preds = model.predict(X_dev)
    rmse = mean_squared_error(y_dev, y_preds, squared=False)
    print("The accuracy for", col, "is", r_sq, "RMSE:", rmse)

    model = LinearRegression()
    model.fit(X_train, y_train)
    r_sq = model.score(X_test, y_test)
    y_preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_preds, squared=False)
    print("The accuracy for", col, "is", r_sq, "RMSE:", rmse)





main()
