""" This module contains unit tests for the linear regression
algorithm """
import unittest
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from machine_learning_algorithms.supervised_learning.regression.linear_regression import LinearRegression
from machine_learning_algorithms.supervised_learning.regression.linear_regression import LassoRegression
from machine_learning_algorithms.supervised_learning.regression.linear_regression import RidgeRegression
from machine_learning_algorithms.utility.ScoreFunctions import RMSE, R_squared


class TestLinearRegression(unittest.TestCase):
    """ This class contains unit tests for the linear regression
    algorithm """

    def setUp(self) -> None:
        x, y = sklearn.datasets.load_boston(return_X_y=True)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.10, random_state=42)

        self.x_train = preprocessing.scale(self.x_train)
        self.x_test = preprocessing.scale(self.x_test).T
        self.y_test = self.y_test.T.reshape(1, -1)
        self.x_train, self.x_valid, self.y_train, self.y_valid = (
            train_test_split(self.x_train,
                             self.y_train,
                             test_size=0.10,
                             random_state=42))
        self.x_train = self.x_train.T
        self.x_valid = self.x_valid.T
        self.y_train = self.y_train.T.reshape(1, -1)
        self.y_valid = self.y_valid.T.reshape(1, -1)
        return super().setUp()

    def test1(self) -> None:
        lr_obj = LinearRegression(degree=1)
        lr_obj.fit_iterative_optimizer(xtrain=self.x_train,
                                       ytrain=self.y_train,
                                       xvalid=self.x_valid,
                                       yvalid=self.y_valid,
                                       num_epochs=100,
                                       ret_train_loss=True,
                                       learn_rate=0.1)
        preds = lr_obj.predict_linear_regression(self.x_test)
        r_squared_val = R_squared(self.y_test, preds)
        self.assertGreaterEqual(r_squared_val, 0.5)

    def test2(self):
        lasso_obj = LassoRegression(degree=1, regParam=1000)
        lasso_obj.fit_iterative_optimizer(xtrain=self.x_train,
                                          ytrain=self.y_train,
                                          xvalid=self.x_valid,
                                          yvalid=self.y_valid,
                                          num_epochs=100,
                                          ret_train_loss=True,
                                          learn_rate=0.1)
        preds = lasso_obj.predict_linear_regression(self.x_test)
        r_squared_val = R_squared(self.y_test, preds)
        self.assertGreaterEqual(r_squared_val, 0.5)

    def test3(self):
        ridge_obj = RidgeRegression(degree=1, regParam=1000)
        ridge_obj.fit_iterative_optimizer(xtrain=self.x_train,
                                          ytrain=self.y_train,
                                          xvalid=self.x_valid,
                                          yvalid=self.y_valid,
                                          num_epochs=100,
                                          ret_train_loss=True,
                                          learn_rate=0.1)
        preds2 = ridge_obj.predict_linear_regression(self.x_test)
        r_squared_val = R_squared(self.y_test, preds2)
        self.assertGreaterEqual(r_squared_val, 0.5)

    def test4(self):
        lin_reg = sklearn.linear_model.LinearRegression()
        lin_reg.fit(self.x_train.T, self.y_train.ravel())
        preds_linreg = lin_reg.predict(self.x_test.T)
        r_squared_sk = R_squared(self.y_test, preds_linreg)

        lin_regOwn = LinearRegression(degree=1)
        lin_regOwn.fit_iterative_optimizer(xtrain=self.x_train,
                                           ytrain=self.y_train,
                                           num_epochs=50,
                                           learn_rate=0.15)
        preds_lrOwn = lin_regOwn.predict_linear_regression(self.x_test)
        r_squared_own = R_squared(self.y_test, preds_lrOwn)

        self.assertLessEqual(abs(r_squared_own - r_squared_sk), 0.07)

    def test5(self):
        lasso_sk = sklearn.linear_model.Lasso(alpha=1)
        lasso_sk.fit(self.x_train.T, self.y_train.ravel())
        preds_lassosk = lasso_sk.predict(self.x_test.T)
        r_squared_sk = R_squared(self.y_test, preds_lassosk)

        lasso_obj2 = LassoRegression(degree=1, regParam=1)
        lasso_obj2.fit_iterative_optimizer(xtrain=self.x_train,
                                           ytrain=self.y_train,
                                           num_epochs=15,
                                           learn_rate=0.15)
        preds_lasso = lasso_obj2.predict_linear_regression(self.x_test)
        r_squared_own = R_squared(self.y_test, preds_lasso)
        self.assertLessEqual(abs(r_squared_own - r_squared_sk), 0.07)

    def test6(self):
        ridge_sk = sklearn.linear_model.Ridge(alpha=1000)
        ridge_sk.fit(self.x_train.T, self.y_train.ravel())
        preds_ridgesk = ridge_sk.predict(self.x_test.T)
        print(R_squared(self.y_test, preds_ridgesk))

        ridge_obj2 = RidgeRegression(degree=1, regParam=1000)
        ridge_obj2.fit_iterative_optimizer(xtrain=self.x_train,
                                           ytrain=self.y_train,
                                           num_epochs=200,
                                           learn_rate=0.1)
        preds_ridge = ridge_obj2.predict_linear_regression(self.x_test)
        print(R_squared(self.y_test, preds_ridge))

    def test7(self):
        degree_2 = LinearRegression(degree=2)
        degree_2.fit_iterative_optimizer(xtrain=self.x_train,
                                         ytrain=self.y_train,
                                         num_epochs=275,
                                         learn_rate=0.01,
                                         ret_train_loss=True)
        deg_2 = degree_2.predict_linear_regression(self.x_test)
        r_squared_val = R_squared(self.y_test, deg_2)
        self.assertGreaterEqual(r_squared_val, 0.8)

    def test8(self):
        lasso_estimator = LassoRegression(degree=2, regParam=55)
        lasso_estimator.fit_iterative_optimizer(xtrain=self.x_train,
                                                ytrain=self.y_train,
                                                num_epochs=275,
                                                learn_rate=0.01)
        preds = lasso_estimator.predict_linear_regression(self.x_test)
        r_squared_val = R_squared(self.y_test, preds)
        self.assertGreaterEqual(r_squared_val, 0.85)

    def test9(self):
        ridge_estimator = RidgeRegression(degree=2, regParam=55)
        ridge_estimator.fit_iterative_optimizer(xtrain=self.x_train,
                                                ytrain=self.y_train,
                                                num_epochs=275,
                                                learn_rate=0.01)
        preds = ridge_estimator.predict_linear_regression(self.x_test)
        r_squared_val = R_squared(self.y_test, preds)
        self.assertGreaterEqual(r_squared_val, 0.85)


if __name__ == "__main__":
    unittest.main()
