import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from ML_algorithms.Supervised_Learning.Classifiers.Logistic_Regression import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression as LR
from ML_algorithms.Utility.ScoreFunctions import accuracy
from ML_algorithms.Utility.k_Fold_CV import k_fold_CV
##-- MANUAL TEST W/ Step through debugging----
X, Y = load_breast_cancer(return_X_y=True)

X = preprocessing.scale(X).T
Y = Y.T.reshape(1, -1)

LR1 = LogisticRegression(X.shape[0], classificationThreshold=0.5)

output = k_fold_CV().getKScore(X, Y, accuracy, LR1)

print(output) ## Seems reasonable 

## Function was used to test both OneVAll logistic regression and softmax regression where it worked fine as well 
 