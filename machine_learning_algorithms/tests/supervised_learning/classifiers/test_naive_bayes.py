""" This module contains unit tests for the naive bayes
algorithm """
import unittest
from sklearn.datasets import load_iris, load_wine
from sklearn import preprocessing
from machine_learning_algorithms.supervised_learning.classifiers.gaussian_naive_bayes import GaussianNaiveBayes
from machine_learning_algorithms.utility.score_functions import accuracy
from machine_learning_algorithms.utility.k_fold_cross_validation import KFoldCrossValidation


class TestNaiveBayes(unittest.TestCase):
    """ This class contains unit tests for the naive bayes algorithm"""

    def setUp(self) -> None:
        self.x, self.y = load_iris(return_X_y=True)

        self.x = preprocessing.scale(self.x).T
        self.y = self.y.reshape(1, -1)

        self.x1, self.y1 = load_wine(return_X_y=True)
        self.x1 = self.x1.T
        self.y1 = self.y1.reshape(1, -1)
        self.k_obj = KFoldCrossValidation()
        self.naive_bayes = GaussianNaiveBayes()

        return super().setUp()

    def test1(self):
        # achieves a k-fold validated score of 96.6% -> very good.
        k_score = self.k_obj.get_k_score(self.x, self.y, accuracy,
                                         self.naive_bayes)
        self.assertGreaterEqual(k_score, 0.90)

    def test2(self):
        # achieves a k-fold validated score of 100% -> very good
        k_score = self.k_obj.get_k_score(self.x1, self.y1, accuracy,
                                         self.naive_bayes)
        self.assertGreaterEqual(k_score, 0.90)


if __name__ == "__main__":
    unittest.main()
