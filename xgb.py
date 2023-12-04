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
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb


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
    y = df[['contract']]

    

    X_train, X_t, y_train, y_t = train_test_split(X, y, test_size=0.3, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(X_t, y_t, test_size=0.33, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    ddev = xgb.DMatrix(X_dev, label=y_dev)

    # Define parameters for XGBoost
    params = {
        'objective': 'multi:softmax',  # Multiclass classification
        'num_class': len(np.unique(y_train)),  # Number of classes
        'max_depth': 7,  # Maximum depth of each tree
        'learning_rate': 0.1,  # Step size shrinkage to prevent overfitting
        'subsample': 0.8,  # Fraction of samples used for training each tree
        'colsample_bytree': 0.8,  # Fraction of features used for training each tree
        'eval_metric': 'merror'  # Evaluation metric to monitor (multiclass error rate)
    }



    k_scores = []
    #Loop through k and 
    for k in range(1, 25):
        params = {
            'objective': 'multi:softmax',  # Multiclass classification
            'num_class': len(np.unique(y_train)),  # Number of classes
            'max_depth': k,  # Maximum depth of each tree
            'learning_rate': 0.1,  # Step size shrinkage to prevent overfitting
            'subsample': 0.8,  # Fraction of samples used for training each tree
            'colsample_bytree': 0.8,  # Fraction of features used for training each tree
            'eval_metric': 'merror'  # Evaluation metric to monitor (multiclass error rate)
        }
        model = xgb.train(params, dtrain, 50)

        # Make predictions on the test set
        y_pred = model.predict(ddev)
        accuracy = accuracy_score(y_dev, y_pred)
        k_scores.append(accuracy)

    params = {
        'objective': 'multi:softmax',  # Multiclass classification
        'num_class': len(np.unique(y_train)),  # Number of classes
        'max_depth': 11,  # Maximum depth of each tree
        'learning_rate': 0.1,  # Step size shrinkage to prevent overfitting
        'subsample': 0.8,  # Fraction of samples used for training each tree
        'colsample_bytree': 0.8,  # Fraction of features used for training each tree
        'eval_metric': 'merror'  # Evaluation metric to monitor (multiclass error rate)
    }

    model = xgb.train(params, dtrain, 50)

    y_pred = model.predict(dtest)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

    # model = DecisionTreeClassifier(
    #             max_depth=3,
    #             random_state=42
    #         )
    # model.fit(X_train, y_train)
    # score = model.score(X_test, y_test)
    # preds = model.predict(X_test)
    # print(score)
    # print(np.unique(preds, return_counts=True))
    # print(preds)
    #plot to see clearly
    plt.plot(range(1, 25), k_scores)
    plt.xlabel('Maximum Depth')
    plt.ylabel('Accuracy on Development Set')
    plt.title("Performance of Max Depth on Development Set")
    plt.show()
    

    



main()