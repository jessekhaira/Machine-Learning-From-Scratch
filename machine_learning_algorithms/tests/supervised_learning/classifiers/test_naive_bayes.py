""" This module contains unit tests for the naive bayes
algorithm """
import unittest
from sklearn.datasets import load_iris, load_wine
from sklearn import preprocessing
from machine_learning_algorithms.supervised_learning.classifiers.gaussian_naive_bayes import GaussianNaiveBayes
from machine_learning_algorithms.utility.ScoreFunctions import accuracy
from machine_learning_algorithms.utility.k_Fold_CV import k_fold_CV


class TestNaiveBayes(unittest.TestCase):
    """ This class contains unit tests for the naive bayes algorithm"""

    def setUp(self) -> None:
        self.x, self.y = load_iris(return_X_y=True)

        self.x = preprocessing.scale(self.x).T
        self.y = self.y.reshape(1, -1)

        self.x1, self.y1 = load_wine(return_X_y=True)
        self.x1 = self.x1.T
        self.y1 = self.y1.reshape(1, -1)
        self.k_obj = k_fold_CV()
        self.naive_bayes = GaussianNaiveBayes()

        return super().setUp()

    def test1(self):
        # achieves a k-fold validated score of 96.6% -> very good.
        kScore = self.k_obj.getKScore(self.x, self.y, accuracy,
                                      self.naive_bayes)
        print(kScore)

    def test2(self):
        # achieves a k-fold validated score of 100% -> very good
        kScore = self.k_obj.getKScore(self.x1, self.y1, accuracy,
                                      self.naive_bayes)
        print(kScore)


if __name__ == "__main__":
    unittest.main()
