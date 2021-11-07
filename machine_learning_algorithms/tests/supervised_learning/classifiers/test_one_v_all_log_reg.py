""" This module contains unit tests for the one vs all logistic regression
algorithm """
import unittest
import numpy as np
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import cross_val_score
from machine_learning_algorithms.utility.ScoreFunctions import accuracy
from machine_learning_algorithms.utility.k_Fold_CV import k_fold_CV
from machine_learning_algorithms.supervised_learning.classifiers.logistic_regression import OneVsAllLogisticRegression


class TestOneVAll(unittest.TestCase):

    def setUp(self) -> None:
        self.x, self.y = load_iris(return_X_y=True)

        self.x = preprocessing.scale(self.x).T
        self.y = self.y.T.reshape(1, -1)

        return super().setUp()

    def test1(self):
        num_classes = len(np.unique(self.y))

        OneVAll = OneVsAllLogisticRegression(num_classes,
                                             self.x.shape[0],
                                             num_epochs=450,
                                             learn_rate=0.3)

        crossVal = k_fold_CV()
        kScore = crossVal.getKScore(self.x, self.y, accuracy, OneVAll)

        print(kScore)

        sklearn_log = LR(penalty='none', multi_class='ovr')

        print(cross_val_score(sklearn_log, self.x.T, self.y.ravel()))
