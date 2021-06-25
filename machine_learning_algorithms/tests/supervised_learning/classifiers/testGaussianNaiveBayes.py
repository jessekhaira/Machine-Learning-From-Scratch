from sklearn.datasets import load_iris, load_wine
from sklearn import preprocessing
import unittest
from machine_learning_algorithms.supervised_learning.classifiers.gaussianNaiveBayes import gaussianNaiveBayes
from machine_learning_algorithms.Utility.ScoreFunctions import accuracy
from machine_learning_algorithms.Utility.k_Fold_CV import k_fold_CV

X, Y = load_iris(return_X_y=True)

X = preprocessing.scale(X).T
Y = Y.reshape(1, -1)

X2, Y2 = load_wine(return_X_y=True)
X2 = X2.T
Y2 = Y2.reshape(1, -1)
kObj = k_fold_CV()
naiveBayes = gaussianNaiveBayes()


class tests(unittest.TestCase):

    def test1(self):
        # achieves a k-fold validated score of 96.6% -> very good.
        kScore = kObj.getKScore(X, Y, accuracy, naiveBayes)
        print(kScore)

    def test2(self):
        # achieves a k-fold validated score of 100% -> very good
        kScore = kObj.getKScore(X2, Y2, accuracy, naiveBayes)
        print(kScore)


if __name__ == "__main__":
    unittest.main()
