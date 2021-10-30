""" This module contains unit tests for the logistic regression
algorithm """
import unittest
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression as LR
from machine_learning_algorithms.utility.ScoreFunctions import accuracy
from machine_learning_algorithms.supervised_learning.classifiers.logistic_regression import LogisticRegression


class TestLogisticRegression(unittest.TestCase):
    """ This class contains unit tests for the logistic regression
    algorithm """

    def setUp(self):
        x, y = load_breast_cancer(return_X_y=True)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.33, random_state=42)

        self.x_train = preprocessing.scale(self.x_train)
        self.x_test = preprocessing.scale(self.x_test).T
        self.y_test = self.y_test.T.reshape(1, -1)

        self.x_train, self.x_valid, self.y_train, self.y_valid = (
            train_test_split(self.x_train,
                             self.y_train,
                             test_size=0.15,
                             random_state=42))

        self.x_train = self.x_train.T
        self.x_valid = self.x_valid.T
        self.y_train = self.y_train.T.reshape(1, -1)
        self.y_valid = self.y_valid.T.reshape(1, -1)
        return super().setUp()

    def test1(self):

        obj1 = LogisticRegression(self.x_train.shape[0],
                                  classificationThreshold=0.5,
                                  regularization="L2",
                                  reg_parameter=50)
        _, _, train_acc, valid_acc = obj1.fit(self.x_train,
                                              self.y_train,
                                              self.x_valid,
                                              self.y_valid,
                                              ret_train_loss=True,
                                              num_epochs=100,
                                              learn_rate=0.1)

        train_acc = np.average(train_acc)
        valid_acc = np.average(valid_acc)

        self.assertGreaterEqual(train_acc, 0.95)
        self.assertGreaterEqual(valid_acc, 0.95)

    def test2(self):
        obj2 = LogisticRegression(self.x_train.shape[0],
                                  classificationThreshold=0.5)

        _, _, train_acc, valid_acc = obj2.fit(self.x_train,
                                              self.y_train,
                                              self.x_valid,
                                              self.y_valid,
                                              ret_train_loss=True,
                                              num_epochs=100,
                                              learn_rate=0.1)
        train_acc = np.average(train_acc)
        valid_acc = np.average(valid_acc)

        self.assertGreaterEqual(train_acc, 0.95)
        self.assertGreaterEqual(valid_acc, 0.95)

    def test3(self):

        obj3 = LogisticRegression(self.x_train.shape[0],
                                  classificationThreshold=0.5,
                                  regularization="L1",
                                  reg_parameter=50)

        train_loss3, valid_loss3, train_acc3, valid_acc3 = obj3.fit(
            self.x_train,
            self.y_train,
            self.x_valid,
            self.y_valid,
            ret_train_loss=True,
            num_epochs=100,
            learn_rate=0.1)

    def test4(self):
        sklearn_mode = LR(C=1)
        sklearn_mode.fit(self.x_train.T, self.y_train.ravel())
        preds_sk = sklearn_mode.predict(self.x_test.T)
        (accuracy(self.y_test, preds_sk))

        obj4 = LogisticRegression(self.x_train.shape[0],
                                  classificationThreshold=0.5,
                                  regularization="L2",
                                  reg_parameter=1)
        obj4.fit(self.x_train, self.y_train, num_epochs=50, learn_rate=0.1)
        preds_model = obj4.predict(self.x_test)
        (accuracy(self.y_test, preds_sk))

    def test5(self):
        sklearn_mode = LR(C=0.0021)
        sklearn_mode.fit(self.x_train.T, self.y_train.ravel())
        preds_sk = sklearn_mode.predict(self.x_test.T)
        (accuracy(self.y_test, preds_sk))

        obj5 = LogisticRegression(self.x_train.shape[0],
                                  classificationThreshold=0.5,
                                  regularization="L2",
                                  reg_parameter=476)
        obj5.fit(self.x_train, self.y_train, num_epochs=50, learn_rate=0.1)
        preds_model = obj5.predict(self.x_test)
        (accuracy(self.y_test, preds_sk))


if __name__ == "__main__":
    unittest.main()