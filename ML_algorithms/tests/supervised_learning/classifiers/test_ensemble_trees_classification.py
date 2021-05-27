""" This module contains code for testing the ensemble tree models
for the supervised machine learning task of classification """
from ML_algorithms.Supervised_Learning.Classifiers.bagged_forest_classifier import BaggedForestClassifier
from ML_algorithms.Supervised_Learning.Classifiers.random_forest_classifier import RandomForestClassifier
import unittest
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris

X1, Y1 = load_breast_cancer(return_X_y=True)

X1 = X1.T
Y1 = Y1.T.reshape(1, -1)

X2, Y2 = load_iris(return_X_y=True)
X2 = X2.T
Y2 = Y2.T.reshape(1, -1)


class TestEnsembleTreesClassification(unittest.TestCase):
    """ This class contains unit tests for ensemble tree models
    performing classification """

    def testBinaryRF(self):
        # Use 5 random bootstrapped samples to train each tree and then get OOB acc and error
        # should be quite low
        mod2 = RandomForestClassifier(criterion="entropy",
                                      verbose=True,
                                      bootstrap=True,
                                      max_samples=5,
                                      minSamplesSplit=1,
                                      maxFeatures=int(X1.shape[0]**0.5))
        mod2.fit(X1, Y1)
        acc2, err2 = mod2.get_oob_score(X1, Y1)
        print(acc2, err2)
        self.assertLessEqual(acc2, 0.90)
        self.assertLessEqual(err2, 0.25)

        # fit on a slightly bigger portion and see what happens - should be better than before
        mod4 = RandomForestClassifier(criterion="entropy",
                                      verbose=True,
                                      bootstrap=True,
                                      max_samples=15,
                                      minSamplesSplit=3,
                                      maxFeatures=int(X1.shape[0]**0.5))
        mod4.fit(X1, Y1)
        acc4, err4 = mod4.get_oob_score(X1, Y1)
        print(acc4, err4)
        self.assertGreaterEqual(acc4, acc2)
        self.assertLessEqual(err4, err2)

        # allow each tree to see a large bootstrapped sample and get accuracy and error - should performed quite well
        # may not perform as well as just one decision tree on this particular dataset since the training procedure has so much stochasiticy in it
        # and it is already difficult to overfit to this dataset
        mod5 = RandomForestClassifier(criterion="entropy",
                                      verbose=True,
                                      bootstrap=True,
                                      max_samples=150,
                                      minSamplesSplit=25,
                                      maxFeatures=int(X1.shape[0]**0.5),
                                      maxDepth=4,
                                      min_impurity_decrease=0.15)
        mod5.fit(X1, Y1)
        acc5, err5 = mod5.get_oob_score(X1, Y1)
        print(acc5, err5)
        self.assertGreaterEqual(acc5, acc4)
        self.assertLessEqual(err5, err4)

    def testMultiClassRF(self):
        # Use 5 random bootstrapped samples to train each tree and then get OOB acc and error
        # should be quite low
        mod2 = RandomForestClassifier(criterion="entropy",
                                      verbose=True,
                                      bootstrap=True,
                                      max_samples=5,
                                      minSamplesSplit=1,
                                      maxFeatures=int(X2.shape[0]**0.5))
        mod2.fit(X2, Y2)
        acc2, err2 = mod2.get_oob_score(X2, Y2)
        print(acc2, err2)
        self.assertLessEqual(acc2, 0.35)
        self.assertGreaterEqual(err2, 0.6)

        # allow more samples into each bootstrapped sample and test - should be better than before

        mod4 = RandomForestClassifier(criterion="entropy",
                                      verbose=True,
                                      bootstrap=True,
                                      max_samples=35,
                                      minSamplesSplit=10,
                                      maxFeatures=int(X2.shape[0]**0.5))
        mod4.fit(X2, Y2)
        acc4, err4 = mod4.get_oob_score(X2, Y2)
        print(acc4, err4)
        self.assertGreaterEqual(acc4, acc2)
        self.assertLessEqual(err4, err2)

        # allow each tree to see a large bootstrapped sample and get accuracy and error - should performed quite well
        # may not perform as well as just one decision tree on this particular dataset since the training procedure has so much stochasiticy in it
        # and it is already difficult to overfit to this dataset
        mod5 = RandomForestClassifier(criterion="entropy",
                                      verbose=True,
                                      bootstrap=True,
                                      max_samples=100,
                                      minSamplesSplit=20,
                                      maxFeatures=int(X2.shape[0]**0.5))
        mod5.fit(X2, Y2)
        acc5, err5 = mod5.get_oob_score(X2, Y2)
        print(acc5, err5)
        self.assertGreaterEqual(acc5, acc4)
        self.assertLessEqual(err5, err4)

    def testBaggedForestBinary(self):
        ## Diff b/w bagged forest and random forest - bagged forests sees every single feature at every single split point
        # can achieve lower bias since there is naturally less stochasiticy in the training procedure, but will struggle with overfitting
        # and lack of variety in the ensemble
        mod2 = BaggedForestClassifier(criterion="entropy",
                                      verbose=True,
                                      bootstrap=True,
                                      max_samples=5,
                                      minSamplesSplit=1)
        mod2.fit(X1, Y1)
        acc2, err2 = mod2.get_oob_score(X1, Y1)
        print(acc2, err2)
        self.assertLessEqual(acc2, 0.90)
        self.assertGreaterEqual(err2, 0.05)

        # fit on a slightly bigger portion and see what happens - should be better than before
        mod4 = BaggedForestClassifier(criterion="entropy",
                                      verbose=True,
                                      bootstrap=True,
                                      max_samples=15,
                                      minSamplesSplit=3)
        mod4.fit(X1, Y1)
        acc4, err4 = mod4.get_oob_score(X1, Y1)
        print(acc4, err4)
        self.assertGreaterEqual(acc4, acc2)
        self.assertLessEqual(err4, err2)

        # allow each tree to see a large bootstrapped sample and get accuracy and error - should performed quite well
        # may not perform as well as just one decision tree on this particular dataset since the training procedure has so much stochasiticy in it
        # and it is already difficult to overfit to this dataset
        mod5 = BaggedForestClassifier(criterion="entropy",
                                      verbose=True,
                                      bootstrap=True,
                                      max_samples=150,
                                      minSamplesSplit=20)
        mod5.fit(X1, Y1)
        acc5, err5 = mod5.get_oob_score(X1, Y1)
        print(acc5, err5)
        self.assertGreaterEqual(acc5, acc4)
        self.assertLessEqual(err5, err4)

    def testBaggedForestMutliclass(self):
        ## Diff b/w bagged forest and random forest - bagged forests sees every single feature at every single split point
        # can achieve lower bias since there is naturally less stochasiticy in the training procedure, but will struggle with overfitting
        # and lack of variety in the ensemble
        mod2 = BaggedForestClassifier(criterion="gini",
                                      verbose=True,
                                      bootstrap=True,
                                      max_samples=5,
                                      minSamplesSplit=1)
        mod2.fit(X2, Y2)
        acc2, err2 = mod2.get_oob_score(X2, Y2)
        print(acc2, err2)
        self.assertGreaterEqual(acc2, 0.70)
        self.assertGreaterEqual(err2, 0.10)

        # allow more samples into each bootstrapped sample and test - should be better than before

        mod4 = BaggedForestClassifier(criterion="entropy",
                                      verbose=True,
                                      bootstrap=True,
                                      max_samples=35,
                                      minSamplesSplit=20)
        mod4.fit(X2, Y2)
        acc4, err4 = mod4.get_oob_score(X2, Y2)
        print(acc4, err4)
        self.assertGreaterEqual(acc4, acc2)
        self.assertLessEqual(err4, err2)

        # allow each tree to see a large bootstrapped sample and get accuracy and error - should performed quite well
        # may not perform as well as just one decision tree on this particular dataset since the training procedure has so much stochasiticy in it
        # and it is already difficult to overfit to this dataset
        mod5 = BaggedForestClassifier(criterion="gini",
                                      verbose=True,
                                      bootstrap=True,
                                      max_samples=100,
                                      minSamplesSplit=20)
        mod5.fit(X2, Y2)
        acc5, err5 = mod5.get_oob_score(X2, Y2)
        print(acc5, err5)
        self.assertGreaterEqual(acc5, acc4)
        self.assertLessEqual(err5, err4)


if __name__ == "__main__":
    unittest.main()