""" This module contains tests for the classification tree algorithm """
import unittest
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from ML_algorithms.Supervised_Learning.Classifiers.classification_tree import ClassificationTree
from ML_algorithms.Utility.ScoreFunctions import accuracy
from ML_algorithms.Utility.k_Fold_CV import k_fold_CV

X1, Y1 = load_breast_cancer(return_X_y=True)
X1 = X1.T
Y1 = Y1.T.reshape(1, -1)

X2, Y2 = load_iris(return_X_y=True)
X2 = X2.T
Y2 = Y2.T.reshape(1, -1)
kCV = k_fold_CV()


class TestsClassificationTree(unittest.TestCase):
    """ This class contains unit tests for the classification tree algorithm """

    def testMultiClass(self):
        # Should be able to overfit multiiclasses easy
        # notice lack of preprocessing - no need to normalize features
        # no need to one hot encode labels

        # out of box model :D
        classificationObj = ClassificationTree(entropy=False, minSamplesSplit=1)
        classificationObj.fit(X2, Y2)
        predictions = classificationObj.predict(X2)
        acc = accuracy(Y2, predictions)
        self.assertEqual(acc, 1)

        classificationObj2 = ClassificationTree(entropy=False,
                                                minSamplesSplit=1)
        kScore = kCV.getKScore(X2, Y2, accuracy, classificationObj2)
        self.assertEqual(kScore, 1)

    def testBinary(self):
        # The tree we are growing is unconstrained so it should be able to
        # fit the training set perfectly 100% (AKA overfitting  :) )
        classificationObj = ClassificationTree(entropy=False, minSamplesSplit=1)
        classificationObj.fit(X1, Y1)
        predictions = classificationObj.predict(X1)
        acc = accuracy(Y1, predictions)
        self.assertEqual(acc, 1)
        classificationObj2 = ClassificationTree(entropy=False,
                                                minSamplesSplit=1)
        kScore = kCV.getKScore(X1, Y1, accuracy, classificationObj2)
        self.assertEqual(kScore, 1)

    def test_sanityChecks(self):
        classificationObj2 = ClassificationTree(entropy=False,
                                                minSamplesSplit=5,
                                                maxDepth=3,
                                                min_impurity_decrease=0.09)
        classificationObj2.fit(X1, Y1)
        predictions2 = classificationObj2.predict(X1)
        acc2 = accuracy(Y1, predictions2)

        classificationObj3 = ClassificationTree(entropy=False,
                                                minSamplesSplit=10,
                                                maxDepth=0,
                                                min_impurity_decrease=0.15)
        classificationObj3.fit(X1, Y1)
        predictions3 = classificationObj3.predict(X1)
        acc3 = accuracy(Y1, predictions3)

        classificationObj4 = ClassificationTree(entropy=False,
                                                minSamplesSplit=1000,
                                                maxDepth=10,
                                                min_impurity_decrease=0.15)
        classificationObj4.fit(X1, Y1)
        predictions4 = classificationObj4.predict(X1)
        acc4 = accuracy(Y1, predictions4)

        classificationObj5 = ClassificationTree(entropy=False,
                                                minSamplesSplit=10,
                                                maxDepth=10,
                                                min_impurity_decrease=1)
        classificationObj5.fit(X1, Y1)
        predictions5 = classificationObj4.predict(X1)
        acc5 = accuracy(Y1, predictions5)

        self.assertLessEqual(acc2, 1)
        self.assertAlmostEqual(acc3, 0.6274165202108963)
        self.assertAlmostEqual(acc4, 0.6274165202108963)
        self.assertAlmostEqual(acc5, 0.6274165202108963)


if __name__ == "__main__":
    unittest.main()
