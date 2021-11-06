""" This module contains unit tests for the one vs all logistic regression
algorithm """
import unittest
import numpy as np
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression as LR
from machine_learning_algorithms.utility.ScoreFunctions import accuracy
from machine_learning_algorithms.utility.k_Fold_CV import k_fold_CV
from machine_learning_algorithms.supervised_learning.classifiers.logistic_regression import OneVsAllLogisticRegression
from sklearn.model_selection import cross_val_score

##-- MANUAL TEST W/ Step through debugging----
X, Y = load_iris(return_X_y=True)

X = preprocessing.scale(X).T
Y = Y.T.reshape(1, -1)

num_classes = len(np.unique(Y))

# In order to train the weights for every logistic regression
# model, you have to train for a tonne of epochs -
# training for 10 epochs gets cross val score of 0.78

# Training for 350 epochs gets ~0.95. 450 epochs ~0.96

OneVAll = OneVsAllLogisticRegression(num_classes,
                                     X.shape[0],
                                     num_epochs=450,
                                     learn_rate=0.3)

crossVal = k_fold_CV()
kScore = crossVal.getKScore(X, Y, accuracy, OneVAll)

print(kScore)

sklearn_log = LR(penalty='none', multi_class='ovr')

print(cross_val_score(sklearn_log, X.T, Y.ravel()))
"""
My OneVAll log reg achieves a cross val score of 0.96 by training for 450 epochs with a learning rate of 0.3, which 
comparable to sklearns implementation of this model. 

KEY thing learned: These tests showed the importance of the hyperparameter learning rate AND the number of epochs you want to train for on
the results. 
Setting learning rate to 0.01 -> awful results. Cross validation score of 81%. The parameters just update to slow for each model, and we have a lot
of models to train. 
Learning rate 0.15 -> Better results. Cross val score 93%
Learning rate 0.3 -> 0.96% cross val score. 
"""
