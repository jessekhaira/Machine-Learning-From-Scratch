""" This module contains code that tests the k fold cv algorithm """
from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer
from ML_algorithms.Supervised_Learning.Classifiers.Logistic_Regression import LogisticRegression
from ML_algorithms.Utility.ScoreFunctions import accuracy
from ML_algorithms.Utility.k_Fold_CV import k_fold_CV
##-- MANUAL TEST W/ Step through debugging----
X, Y = load_breast_cancer(return_X_y=True)

X = preprocessing.scale(X).T
Y = Y.T.reshape(1, -1)

LR1 = LogisticRegression(X.shape[0], classificationThreshold=0.5)

output = k_fold_CV().getKScore(X, Y, accuracy, LR1)

print(output)    ## Seems reasonable
