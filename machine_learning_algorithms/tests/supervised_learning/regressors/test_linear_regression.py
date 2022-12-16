""" This module contains unit tests for the linear regression
algorithm """
import numpy as np
import unittest
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
from sklearn.linear_model import SGDRegressor as skl_lr
from sklearn.linear_model import LinearRegression as skl_lr_analytic
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from machine_learning_algorithms.supervised_learning.regression.linear_regression import (
    LinearRegression, LassoRegression, RidgeRegression)
from machine_learning_algorithms.utility.score_functions import r_squared
from machine_learning_algorithms.utility.misc import rel_error


class TestLinearRegression(unittest.TestCase):
    """ This class contains unit tests for the linear regression
    algorithm """

    def setUp(self) -> None:
        x, y = sklearn.datasets.load_diabetes(return_X_y=True)

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
                                       num_epochs=500,
                                       ret_train_loss=True,
                                       learn_rate=0.01)
        preds = lr_obj.predict_linear_regression(self.x_test)
        r_squared_val = r_squared(self.y_test, preds)

        sklearn_lr_obj = skl_lr(penalty=None,
                                learning_rate='constant',
                                eta0=0.01,
                                max_iter=500)
        sklearn_lr_obj.fit(self.x_train.T, self.y_train.T)
        preds_skl = sklearn_lr_obj.predict(self.x_test.T).reshape(1, -1)
        r_squared_val_skl = r_squared(self.y_test, preds_skl.T)

        self.assertTrue(r_squared_val >= r_squared_val_skl)

    def test_poly_features(self) -> None:
        poly = preprocessing.PolynomialFeatures(degree=2, include_bias=False)
        lr_obj = LinearRegression(2)
        x_poly_train = poly.fit_transform(self.x_train.T)
        lr_data_poly = lr_obj._get_polynomial_features(self.x_train)
        self.assertEqual(x_poly_train.T.shape, lr_data_poly.shape)

    def test2_polynominal(self) -> None:
        lr_obj = LinearRegression(degree=2)
        lr_obj.fit_iterative_optimizer(xtrain=self.x_train,
                                       ytrain=self.y_train,
                                       num_epochs=500,
                                       ret_train_loss=True,
                                       learn_rate=0.01)
        preds = lr_obj.predict_linear_regression(self.x_test)
        r_squared_val = r_squared(self.y_test, preds)

        poly = preprocessing.PolynomialFeatures(degree=2, include_bias=False)
        x_poly_trskl = poly.fit_transform(self.x_train.T)
        x_poly_teskl = poly.fit_transform(self.x_test.T)

        sklearn_lr_obj = skl_lr(penalty=None, max_iter=500, eta0=0.01)
        sklearn_lr_obj.fit(x_poly_trskl, self.y_train.T)
        preds_skl = sklearn_lr_obj.predict(x_poly_teskl).reshape(1, -1)
        assert (self.y_test.shape == preds_skl.shape)
        r_squared_val_skl = r_squared(self.y_test, preds_skl)
        self.assertTrue(r_squared_val >= r_squared_val_skl)


if __name__ == "__main__":
    unittest.main()
