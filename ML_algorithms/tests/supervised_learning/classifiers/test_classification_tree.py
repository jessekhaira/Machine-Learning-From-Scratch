""" This module contains tests for the classification tree algorithm """
import unittest
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from ML_algorithms.Supervised_Learning.Classifiers.classification_tree import ClassificationTree
from ML_algorithms.Utility.ScoreFunctions import accuracy
from ML_algorithms.Utility.k_Fold_CV import k_fold_CV


class TestsClassificationTree(unittest.TestCase):
    """ This class contains unit tests for the classification tree algorithm """

    def setUp(self):
        self.x1, self.y1 = load_breast_cancer(return_X_y=True)
        self.x1 = self.x1.T
        self.y1 = self.y1.T.reshape(1, -1)

        self.x2, self.y2 = load_iris(return_X_y=True)
        self.x2 = self.x2.T
        self.y2 = self.y2.T.reshape(1, -1)
        self.k_cv = k_fold_CV()

    def tearDown(self):
        self.x1 = None
        self.y1 = None
        self.x2 = None
        self.y2 = None
        self.k_cv = None

    def test_multi_class(self):
        # Should be able to overfit multiiclasses easy
        # notice lack of preprocessing - no need to normalize features
        # no need to one hot encode labels

        # out of box model :D
        classification_obj = ClassificationTree(entropy=False,
                                                minSamplesSplit=1)
        classification_obj.fit(self.x2, self.y2)
        predictions = classification_obj.predict(self.x2)
        acc = accuracy(self.y2, predictions)
        self.assertEqual(acc, 1)

        classification_obj2 = ClassificationTree(entropy=False,
                                                 minSamplesSplit=1)
        kScore = self.k_cv.getKScore(self.x2, self.y2, accuracy,
                                     classification_obj2)
        self.assertEqual(kScore, 1)

    def test_binary(self):
        # The tree we are growing is unconstrained so it should be able to
        # fit the training set perfectly 100% (AKA overfitting  :) )
        classification_obj = ClassificationTree(entropy=False,
                                                minSamplesSplit=1)
        classification_obj.fit(self.x1, self.y1)
        predictions = classification_obj.predict(self.x1)
        acc = accuracy(self.y1, predictions)
        self.assertEqual(acc, 1)
        classification_obj2 = ClassificationTree(entropy=False,
                                                 minSamplesSplit=1)
        kScore = self.k_cv.getKScore(self.x1, self.y1, accuracy,
                                     classification_obj2)
        self.assertEqual(kScore, 1)

    def test_sanity_checks(self):
        classification_obj2 = ClassificationTree(entropy=False,
                                                 minSamplesSplit=5,
                                                 maxDepth=3,
                                                 min_impurity_decrease=0.09)
        classification_obj2.fit(self.x1, self.y1)
        predictions2 = classification_obj2.predict(self.x1)
        acc2 = accuracy(self.y1, predictions2)

        classification_obj3 = ClassificationTree(entropy=False,
                                                 minSamplesSplit=10,
                                                 maxDepth=0,
                                                 min_impurity_decrease=0.15)
        classification_obj3.fit(self.x1, self.y1)
        predictions3 = classification_obj3.predict(self.x1)
        acc3 = accuracy(self.y1, predictions3)

        classification_obj4 = ClassificationTree(entropy=False,
                                                 minSamplesSplit=1000,
                                                 maxDepth=10,
                                                 min_impurity_decrease=0.15)
        classification_obj4.fit(self.x1, self.y1)
        predictions4 = classification_obj4.predict(self.x1)
        acc4 = accuracy(self.y1, predictions4)

        classification_obj5 = ClassificationTree(entropy=False,
                                                 minSamplesSplit=10,
                                                 maxDepth=10,
                                                 min_impurity_decrease=1)
        classification_obj5.fit(self.x1, self.y1)
        predictions5 = classification_obj4.predict(self.x1)
        acc5 = accuracy(self.y1, predictions5)

        self.assertLessEqual(acc2, 1)
        self.assertAlmostEqual(acc3, 0.6274165202108963)
        self.assertAlmostEqual(acc4, 0.6274165202108963)
        self.assertAlmostEqual(acc5, 0.6274165202108963)


if __name__ == "__main__":
    unittest.main()
