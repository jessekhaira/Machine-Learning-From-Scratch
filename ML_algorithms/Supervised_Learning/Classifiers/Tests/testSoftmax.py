import sys
sys.path.append("/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Supervised_Learning/Classifiers")
sys.path.append("/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Utility")
sys.path.append("/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Neural_Net_Util")
from SoftmaxRegression import softmax_regression
from Optimizers import Adam
from Optimizers import RMSProp
import unittest
import numpy as np 
import sklearn
from ScoreFunctions import accuracy
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from misc import oneHotEncode
from k_Fold_CV import k_fold_CV


"""
The implementation of the softmax classifier is correct. With 8 folds, a cross validation score
of 94.4% accuracy was achieved. With L1 regularization and L2 regularization, the classifier 
performs as expected -> performance is very sensitive to regParameter. If the regularization parameter is even slightly high (>0.3),
the performance for the l1 regularized and l2 regularized softmax regression models falter heavily. 
"""

##-- MANUAL TEST W/ Step through debugging----
X, Y = load_iris(return_X_y=True)

X = preprocessing.scale(X).T

y_encoded = oneHotEncode(Y)

softmaxReg = softmax_regression(X.shape[0], len(y_encoded))

softmaxReg1 = softmax_regression(X.shape[0], len(y_encoded), regularization="L1", regParameter=0.01)

softmaxReg2 = softmax_regression(X.shape[0], len(y_encoded), regularization="L2", regParameter=0.01)

## Strength of RMSProp shown - get a 6% increase in accuracy w/ it. 99.3% RMSprop and 93.7% normal gradient descent 
objKfold = k_fold_CV()
kScore_normalGD = objKfold.getKScore(X, y_encoded, accuracy, softmaxReg, numEpochs = 100, learn_rate = 0.2, k=8)
print(kScore_normalGD)

kScore_RMSProp = objKfold.getKScore(X, y_encoded, accuracy, softmaxReg, numEpochs = 100, learn_rate = 0.2, k=8, optim = RMSProp())
print(kScore_RMSProp)

# Adam is the most sensitive out of the three tested and requires the most hyperparameter tuning
train_loss, train_acc = softmaxReg.fit(X, y_encoded, num_epochs=1000, learn_rate=0.01, optim=Adam(), ret_train_loss=True)
kScore_Adam = objKfold.getKScore(X, y_encoded, accuracy, softmaxReg, numEpochs=1000, learn_rate=0.01, k=8, optim=Adam())
print(kScore_Adam)
print(train_loss)
print(train_acc)

kScore1 = objKfold.getKScore(X, y_encoded, accuracy, softmaxReg1, numEpochs = 150, learn_rate = 0.01, k=8)
print(kScore1)

kScore2 = objKfold.getKScore(X, y_encoded, accuracy, softmaxReg2, numEpochs = 150, learn_rate = 0.01, k=8)
print(kScore2)

