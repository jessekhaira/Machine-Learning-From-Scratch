from machine_learning_algorithms.supervised_learning.regression.linear_regression import LinearRegression
from machine_learning_algorithms.supervised_learning.regression.linear_regression import LassoRegression
from machine_learning_algorithms.supervised_learning.regression.linear_regression import RidgeRegression
from machine_learning_algorithms.Utility.ScoreFunctions import RMSE, R_squared
import unittest
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

### ---  MANUAL TESTING W/ Step through Debugging --- ###
X, Y = sklearn.datasets.load_boston(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.10,
                                                    random_state=42)

X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test).T
y_test = y_test.T.reshape(1, -1)

X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                      y_train,
                                                      test_size=0.10,
                                                      random_state=42)

X_train = X_train.T
X_valid = X_valid.T
y_train = y_train.T.reshape(1, -1)
y_valid = y_valid.T.reshape(1, -1)
"""
To sanity check the implementations, the regularization parameter was cranked up to 1000. With this level for the 
regularization parameter, we should see a couple things:

- L1 regularization sets the weights for some features really low, other really high. This is why L1 regularization
is said to be good for feature selection. The model can learn which features should be heavily weighted, and which
should recieve no weight at all.
- L2 regularization should decay most of the weights to around zero, but none should be set as low L1 regularization.
So we still use every input dimension in our output prediction. 
- The costs for the regularized models should be really high. 
- The l1 and l2 norms of the regularized models should be MUCH lower then the l1 and l2 norms of the linear model.

All observations held. The models were comparable when the regularization parameter was low, say set equal to 0.1, except
the regularized models generalized better than the normal model. 

I played around with sklearns implementation of lasso and ridge regression as well to compare my model to these as a label
and I got pretty much the exact same R2 score when trained on the same data w/ the same regularization paramters 
(after playing around with the learning rate and num_epochs to train for doing batch gradient descent). 

Then I went and fit higher capacity polynomial regression models and improved my r2 score overall! 
"""

## Degree == 1 Polynomial
lr_obj = LinearRegression(degree=1)
lasso_obj = LassoRegression(degree=1, regParam=1000)
ridge_obj = RidgeRegression(degree=1, regParam=1000)

print('\n')
train_loss1, valid_loss1, train_acc1, valid_acc1 = lr_obj.fit_iterative_optimizer(
    xtrain=X_train,
    ytrain=y_train,
    xvalid=X_valid,
    yvalid=y_valid,
    num_epochs=100,
    ret_train_loss=True,
    learn_rate=0.1)
print(train_loss1, valid_loss1)
print(lr_obj.layers[0].W.T)
preds = lr_obj.predict_linear_regression(X_test)
print(R_squared(y_test, preds))
print(np.linalg.norm(lr_obj.layers[0].W)**2)
print('\n')

train_loss2, valid_loss2, train_acc2, valid_acc2 = lasso_obj.fit_iterative_optimizer(
    xtrain=X_train,
    ytrain=y_train,
    xvalid=X_valid,
    yvalid=y_valid,
    num_epochs=100,
    ret_train_loss=True,
    learn_rate=0.1)
print(train_loss2, valid_loss2)
print(lasso_obj.layers[0].W.T)
preds1 = lasso_obj.predict_linear_regression(X_test)
print(R_squared(y_test, preds1))
print(np.linalg.norm(lasso_obj.layers[0].W, ord=1))
print(np.linalg.norm(lasso_obj.layers[0].W)**2)
print('\n')

train_loss3, valid_loss3, train_acc3, valid_acc3 = ridge_obj.fit_iterative_optimizer(
    xtrain=X_train,
    ytrain=y_train,
    xvalid=X_valid,
    yvalid=y_valid,
    num_epochs=100,
    ret_train_loss=True,
    learn_rate=0.1)
print(train_loss3, valid_loss3)
print(ridge_obj.layers[0].W.T)
preds2 = ridge_obj.predict_linear_regression(X_test)
print(R_squared(y_test, preds2))
print(np.linalg.norm(ridge_obj.layers[0].W, ord=1))
print(np.linalg.norm(ridge_obj.layers[0].W, ord=2)**2)
print('\n')

## Linear Reg ##

##SKLEARN LASSO##
# Training for 100 epochs w/ a learning rate of 0.15 gets the exact same R2 score between the models.
# With a bit hyperparameter tuning, only training for 50 epochs with a learning rate of 0.15, the
# R2 score performance increases and my model outperforms the sklearn model slightly (0.692 to 0.687).
lin_reg = sklearn.linear_model.LinearRegression()
lin_reg.fit(X_train.T, y_train.ravel())
preds_linreg = lin_reg.predict(X_test.T)
print(R_squared(y_test, preds_linreg))

lin_regOwn = LinearRegression(degree=1)
lin_regOwn.fit_iterative_optimizer(xtrain=X_train,
                                   ytrain=y_train,
                                   num_epochs=50,
                                   learn_rate=0.15)
preds_lrOwn = lin_regOwn.predict_linear_regression(X_test)
print(R_squared(y_test, preds_lrOwn))

print('\n')

# These models are estimated differently and hence can't be compared exactly
# but with a reg paramter of 1 and training for 15 epochs at a leaerning rate of 15,
# my implementation out performs the implementation from sklearn by a significant margin: 0.703 sklearn, 0.757 own
lasso_sk = sklearn.linear_model.Lasso(alpha=1)
lasso_sk.fit(X_train.T, y_train.ravel())
preds_lassosk = lasso_sk.predict(X_test.T)
print(R_squared(y_test, preds_lassosk))

lasso_obj2 = LassoRegression(degree=1, regParam=1)
lasso_obj2.fit_iterative_optimizer(xtrain=X_train,
                                   ytrain=y_train,
                                   num_epochs=15,
                                   learn_rate=0.15)
preds_lasso = lasso_obj2.predict_linear_regression(X_test)
print(R_squared(y_test, preds_lasso))

print('\n')
##SKLEARN RIDGE##

# setting alpha = 1000 and training for 200 epochs with a learning
# rate of 0.1 gets the exact same R2 score for both models of 0.624. Pretty cool!
ridge_sk = sklearn.linear_model.Ridge(alpha=1000)
ridge_sk.fit(X_train.T, y_train.ravel())
preds_ridgesk = ridge_sk.predict(X_test.T)
print(R_squared(y_test, preds_ridgesk))

ridge_obj2 = RidgeRegression(degree=1, regParam=1000)
ridge_obj2.fit_iterative_optimizer(xtrain=X_train,
                                   ytrain=y_train,
                                   num_epochs=200,
                                   learn_rate=0.1)
preds_ridge = ridge_obj2.predict_linear_regression(X_test)
print(R_squared(y_test, preds_ridge))

## Polynomial Regression ##
# Exact same as other models, except instead of fitting a linear function, we can fit polynomial
# functions with an abritrary degree

# You have to be super careful with the learning rate here or else you will diverge.
print('\n')
degree_2 = LinearRegression(degree=2)
train_loss = degree_2.fit_iterative_optimizer(xtrain=X_train,
                                              ytrain=y_train,
                                              num_epochs=275,
                                              learn_rate=0.01,
                                              ret_train_loss=True)
print(train_loss)
deg_2 = degree_2.predict_linear_regression(X_test)
print(R_squared(y_test, deg_2))

print(RMSE(y_test, deg_2))

print('\n')
lasso_objd2 = LassoRegression(degree=2, regParam=55)
lasso_objd2.fit_iterative_optimizer(xtrain=X_train,
                                    ytrain=y_train,
                                    num_epochs=275,
                                    learn_rate=0.01)
preds_lassod2 = lasso_objd2.predict_linear_regression(X_test)
print(R_squared(y_test, preds_lassod2))
print(RMSE(y_test, preds_lassod2))

print('\n')
ridge_objd2 = RidgeRegression(degree=2, regParam=55)
ridge_objd2.fit_iterative_optimizer(xtrain=X_train,
                                    ytrain=y_train,
                                    num_epochs=275,
                                    learn_rate=0.01)
preds_ridged2 = ridge_objd2.predict_linear_regression(X_test)
print(R_squared(y_test, preds_ridged2))
print(RMSE(y_test, preds_ridged2))
