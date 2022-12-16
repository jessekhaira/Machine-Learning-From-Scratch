""" This module contains unit tests for the linear regression
algorithm """
import unittest
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
from sklearn.linear_model import SGDRegressor as skl_lr
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from machine_learning_algorithms.supervised_learning.regression.linear_regression import (
    LinearRegression, LassoRegression, RidgeRegression)
from machine_learning_algorithms.utility.score_functions import r_squared


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
        sklearn_lr_obj = skl_lr(penalty=None)
        lr_obj.fit_iterative_optimizer(xtrain=self.x_train,
                                       ytrain=self.y_train,
                                       num_epochs=500,
                                       ret_train_loss=True,
                                       learn_rate=0.01)
        sklearn_lr_obj.fit(self.x_train.T, self.y_train.T)
        preds = lr_obj.predict_linear_regression(self.x_test)
        preds_skl = sklearn_lr_obj.predict(self.x_test.T)

        r_squared_val = r_squared(self.y_test, preds)
        r_squared_val_skl = r_squared(self.y_test, preds_skl.T)
        self.assertTrue(abs(r_squared_val - r_squared_val_skl) <= 1e-2)


if __name__ == "__main__":
    unittest.main()
