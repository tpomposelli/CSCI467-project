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

main()