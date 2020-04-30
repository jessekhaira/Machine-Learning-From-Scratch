from ML_algorithms.Supervised_Learning.Classifiers.Logistic_Regression import LogisticRegression
import unittest
import numpy as np 
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression as LR
from ML_algorithms.Utility.ScoreFunctions import accuracy

##-- MANUAL TEST W/ Step through debugging----
X, Y = load_breast_cancer(return_X_y=True)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Standardize train set and test set differently - at inference time, you will not 
# be normalizing your input with the output
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test).T
y_test = y_test.T.reshape(1, -1)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

X_train = X_train.T
X_valid = X_valid.T
y_train = y_train.T.reshape(1, -1)
y_valid = y_valid.T.reshape(1, -1)

obj1 = LogisticRegression(X_train.shape[0], classificationThreshold=0.5)

train_loss, val_loss, train_acc, val_acc = obj1.fit(X_train, y_train, X_valid, y_valid, num_epochs=30, ret_train_loss=True)

print(train_loss)
print('\n')
print(val_loss)
print('\n')
print(train_acc)
print('\n')
print(val_acc)

preds_train = obj1.predict(X_train).T.reshape(1, -1)
preds_test = obj1.predict(X_test).T.reshape(1,-1)

print(accuracy(y_train, preds_train))
print(accuracy(y_test, preds_test))

sklearn_test = LR(penalty = 'none')
sklearn_test.fit(X_train.T, y_train.T)
preds_sklearnTrain = sklearn_test.predict(X_train.T)
preds_sklearnTest = sklearn_test.predict(X_test.T)

print(accuracy(preds_sklearnTrain, y_train))
print(accuracy(preds_sklearnTest, y_test))

"""
Sklearns logistic regression got an accuraccy of 96% on the test set and 100% on the train set.

In contrast, the model implemented by me got an accuracy of 98.9% on the test set and 98.1%
on the training set. 

The implementation looks good!
"""
## -- END MANUAL TEST -- 


