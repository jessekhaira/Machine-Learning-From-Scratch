""" This module contains code that tests the k fold cv algorithm """
from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer
from machine_learning_algorithms.supervised_learning.classifiers.Logistic_Regression import LogisticRegression
from machine_learning_algorithms.utility.ScoreFunctions import accuracy
from machine_learning_algorithms.utility.k_Fold_CV import k_fold_CV
import unittest


class TestKFoldCV(unittest.TestCase):
    """ This class contains unit tests testing the K Fold CV
    algorithm """

    def test_1(self):
        X, Y = load_breast_cancer(return_X_y=True)
        X = preprocessing.scale(X).T
        Y = Y.T.reshape(1, -1)

        LR1 = LogisticRegression(X.shape[0], classificationThreshold=0.5)

        output = k_fold_CV().getKScore(X, Y, accuracy, LR1)

        self.assertGreaterEqual(output, 0.95)


if __name__ == "__main__":
    unittest.main()
