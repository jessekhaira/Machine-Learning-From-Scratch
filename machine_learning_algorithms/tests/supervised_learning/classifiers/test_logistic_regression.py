import unittest
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression as LR
from machine_learning_algorithms.utility.ScoreFunctions import accuracy
from machine_learning_algorithms.supervised_learning.classifiers.logistic_regression import LogisticRegression


class TestLogisticRegression(unittest.TestCase):

    def setUp(self):
        ##-- MANUAL TEST W/ Step through debugging----
        X, Y = load_breast_cancer(return_X_y=True)

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            Y,
                                                            test_size=0.33,
                                                            random_state=42)

        # Standardize train set and test set differently - at inference time, you will not
        # be normalizing your input with the output
        X_train = preprocessing.scale(X_train)
        X_test = preprocessing.scale(X_test).T
        y_test = y_test.T.reshape(1, -1)

        X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                              y_train,
                                                              test_size=0.15,
                                                              random_state=42)

        X_train = X_train.T
        X_valid = X_valid.T
        y_train = y_train.T.reshape(1, -1)
        y_valid = y_valid.T.reshape(1, -1)
        return super().setUp()

    def test1(self):

        obj1 = LogisticRegression(X_train.shape[0],
                                  classificationThreshold=0.5,
                                  regularization="L2",
                                  reg_parameter=50)

    def test2(self):
        obj2 = LogisticRegression(X_train.shape[0], classificationThreshold=0.5)

        train_loss2, valid_loss2, train_acc2, valid_acc2 = obj2.fit(
            X_train,
            y_train,
            X_valid,
            y_valid,
            ret_train_loss=True,
            num_epochs=100,
            learn_rate=0.1)

    def test3(self):

        obj3 = LogisticRegression(X_train.shape[0],
                                  classificationThreshold=0.5,
                                  regularization="L1",
                                  reg_parameter=50)

        train_loss3, valid_loss3, train_acc3, valid_acc3 = obj3.fit(
            X_train,
            y_train,
            X_valid,
            y_valid,
            ret_train_loss=True,
            num_epochs=100,
            learn_rate=0.1)

    def test4(self):
        sklearn_mode = LR(C=1)
        sklearn_mode.fit(X_train.T, y_train.ravel())
        preds_sk = sklearn_mode.predict(X_test.T)
        print(accuracy(y_test, preds_sk))

        obj4 = LogisticRegression(X_train.shape[0],
                                  classificationThreshold=0.5,
                                  regularization="L2",
                                  reg_parameter=1)
        obj4.fit(X_train, y_train, num_epochs=50, learn_rate=0.1)
        preds_model = obj4.predict(X_test)
        print(accuracy(y_test, preds_sk))

    def test5(self):
        sklearn_mode = LR(C=0.0021)
        sklearn_mode.fit(X_train.T, y_train.ravel())
        preds_sk = sklearn_mode.predict(X_test.T)
        print(accuracy(y_test, preds_sk))

        obj5 = LogisticRegression(X_train.shape[0],
                                  classificationThreshold=0.5,
                                  regularization="L2",
                                  reg_parameter=476)
        obj5.fit(X_train, y_train, num_epochs=50, learn_rate=0.1)
        preds_model = obj5.predict(X_test)
        print(accuracy(y_test, preds_sk))


if __name__ == "__main__":
    unittest.main()