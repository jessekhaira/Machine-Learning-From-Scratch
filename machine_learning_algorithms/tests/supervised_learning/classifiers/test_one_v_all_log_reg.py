""" This module contains unit tests for the one vs all logistic regression
algorithm """
import unittest
import numpy as np
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import cross_val_score
from machine_learning_algorithms.utility.score_functions import accuracy
from machine_learning_algorithms.utility.k_Fold_CV import k_fold_CV
from machine_learning_algorithms.supervised_learning.classifiers.logistic_regression import OneVsAllLogisticRegression


class TestOneVAll(unittest.TestCase):
    """ This class contains unit tests for the one vs all
    logistic regression algorithm """

    def setUp(self) -> None:
        self.x, self.y = load_iris(return_X_y=True)

        self.x = preprocessing.scale(self.x).T
        self.y = self.y.T.reshape(1, -1)

        return super().setUp()

    def test1(self):
        """ In order to train the weights for every logistic regression
        model, you have to train for a tonne of epochs.

        Training for 10 epochs gets cross val score of 0.78
        Training for 350 epochs gets ~0.95. 450 epochs ~0.96
        """
        num_classes = len(np.unique(self.y))

        one_vs_all_logistic_regression = OneVsAllLogisticRegression(
            num_classes, self.x.shape[0], num_epochs=450, learn_rate=0.3)

        cross_val = k_fold_CV()
        k_score = cross_val.getKScore(self.x, self.y, accuracy,
                                      one_vs_all_logistic_regression)

        sklearn_log = LR(penalty="none", multi_class="ovr")

        sklearn_score = np.average(
            cross_val_score(sklearn_log, self.x.T, self.y.ravel()))

        difference = abs(sklearn_score - k_score)
        self.assertLessEqual(difference, 0.1)


if __name__ == "__main__":
    unittest.main()
