import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from numpy.linalg import pinv
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


df = pd.read_csv("final_data.csv")


def main():
    X = df.drop(columns={'Player','Tm','Salary S','Season S-1','% of Cap S', 'Salary S-1', 'Cap Maximum S', 'Cap Maximum S-1','% of Cap S-1','% of Cap S'})
    y = df[['% of Cap S']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_list={"penalty": ["l1", "l2"],
            "fit_intercept":[True,False]}
    lg=LogisticRegression(max_iter=1000,solver='liblinear')
    lg_cross_validation=GridSearchCV(lg, param_list, cv=12,scoring='accuracy')
    lg_cross_validation.fit(X_train, y_train)
    print("The best hyperparameters of logistic regression is: {}".format(lg_cross_validation.best_params_))
    print("Best score is {}".format(lg_cross_validation.best_score_))

    penalty=['l1', 'l2']
    fit_intercept=[True,False]
    scores_mean = lg_cross_validation.cv_results_['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(penalty), len(fit_intercept)).T

    print('Best params = {}'.format(lg_cross_validation.best_params_))
    print('Best score = {}'.format(scores_mean.max()))

    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(penalty):
        ax.plot(fit_intercept, scores_mean[idx, :], '-o', label="penalty" + ': ' + str(val))

    ax.tick_params(axis='x', rotation=0)
    ax.set_title('Grid Search Result on Dev Set')
    ax.set_xlabel('fit_intercept')
    ax.set_ylabel('CV score (accuracy)')
    ax.legend(loc='best')
    ax.grid('on')

    confusion_matrix(y_test,lg_cross_validation.predict(X_test))

main()