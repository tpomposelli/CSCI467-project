import pandas as pd
import numpy as np
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import argparse
import sys
from numpy.linalg import pinv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


df = pd.read_csv("final_data.csv")


def main():

    #Set conditions to classify every player season
    conditions = [
        (df['% of Cap S'] <= 4.5) & (df['% of Cap S'] > 0),
        (df['% of Cap S'] <= 12) & (df['% of Cap S'] > 4.5),
        (df['% of Cap S'] <= 24) & (df['% of Cap S'] > 12),
        (df['% of Cap S'] > 24)
    ]
 
    outcomes = [0, 1, 2, 3]

    df['contract'] = np.select(conditions, outcomes, default=0)


    X = df[['PTS_per_game', 'AST_per_game', 'DRB_per_game', 'MP_per_game', 'FG_per_game', 'VORP_advanced', 'STL_per_game', 'TOV_per_game', 'WS_advanced', 'PER_advanced', 'BPM_advanced']]
    y = df[['contract', '% of Cap S']]

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
    k_scores = []
    Loop through k and 
    for k in range(1, 26):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        scores = model.score(X_dev, y_dev)
        k_scores.append(scores)

    print(y_test.to_string())
    y_test = y_test.drop(columns={'% of Cap S'})
    y_train = y_train.drop(columns={'% of Cap S'})

    model = KNeighborsClassifier(n_neighbors=18)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    preds = model.predict(X_test)
    print(score)
    print(np.unique(preds, return_counts=True))
    print(preds)
    #plot to see clearly
    plt.plot(range(1, 26), k_scores)
    plt.xlabel('K-values')
    plt.ylabel('Accuracy on Development Set')
    plt.title("Performance of K-values on Development Set")
    plt.show()
    

    



main()