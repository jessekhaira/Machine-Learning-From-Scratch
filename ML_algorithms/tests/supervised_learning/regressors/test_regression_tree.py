""" This module contains tests for the regression decision tree
algorithm """
import unittest
from sklearn.datasets import load_boston
from ML_algorithms.Supervised_Learning.regression.regression_tree import RegressionTree
from ML_algorithms.Utility.ScoreFunctions import RMSE, MSE, MAE, TSS, RSS
from ML_algorithms.Utility.k_Fold_CV import k_fold_CV


class RegressionTreeTests(unittest.TestCase):
    """ This class contains test for the regression decision tree
    algorithm """

    def setUp(self):
        self.x1, self.y1 = load_boston(return_X_y=True)
        self.x1 = self.x1.T
        self.y1 = self.y1.T.reshape(1, -1)
        self.k_cv = k_fold_CV()

    def tearDown(self):
        self.x1 = None
        self.y1 = None
        self.k_cv = None

    def testFit(self):
        # The tree is not being constrained when it is fitting
        # thus we should be able to achieve 0 error AKA all residuals are 0
        regression_obj = RegressionTree(minSamplesSplit=1)
        regression_obj.fit(self.x1, self.y1)
        predictions = regression_obj.predict(self.x1)
        rmse = RMSE(self.y1, predictions)
        mae = MAE(self.y1, predictions)
        mse = MSE(self.y1, predictions)
        rmse = RMSE(self.y1, predictions)
        self.assertEqual(mae, 0)
        self.assertEqual(mse, 0)
        self.assertEqual(rmse, 0)

        # test generalization of regression tree
        regression_obj2 = RegressionTree(minSamplesSplit=1)
        kScoreRMSE = self.k_cv.getKScore(self.x1, self.y1, RMSE,
                                         regression_obj2)
        kScoreMSE = self.k_cv.getKScore(self.x1, self.y1, MSE, regression_obj2)
        kScoreMAE = self.k_cv.getKScore(self.x1, self.y1, MAE, regression_obj2)
        # Dataset is easy so we should expect 0 error
        self.assertEqual(kScoreRMSE, 0)
        self.assertEqual(kScoreMSE, 0)
        self.assertEqual(kScoreMAE, 0)

    def test_sanityChecks(self):
        # If we regularize the tree, we should get a higher error than if we don't
        regression_obj2 = RegressionTree(minSamplesSplit=5,
                                         maxDepth=3,
                                         min_impurity_decrease=0.15)
        regression_obj2.fit(self.x1, self.y1)
        predictions2 = regression_obj2.predict(self.x1)
        error2 = RMSE(self.y1, predictions2)

        # Sanity checks - regularization is so high all we get is one leaf
        # meaning all predictions are equal to mean of labels
        # meaning that the RSS of the predictions should be equal to TSS of the label
        regression_obj3 = RegressionTree(minSamplesSplit=10,
                                         maxDepth=0,
                                         min_impurity_decrease=0.15)
        regression_obj3.fit(self.x1, self.y1)
        predictions3 = regression_obj3.predict(self.x1)

        regression_obj4 = RegressionTree(minSamplesSplit=1000,
                                         maxDepth=10,
                                         min_impurity_decrease=0.15)
        regression_obj4.fit(self.x1, self.y1)
        predictions4 = regression_obj4.predict(self.x1)

        regression_obj5 = RegressionTree(minSamplesSplit=10,
                                         maxDepth=10,
                                         min_impurity_decrease=1)
        regression_obj5.fit(self.x1, self.y1)
        predictions5 = regression_obj4.predict(self.x1)

        self.assertGreaterEqual(error2, 0)
        self.assertAlmostEqual(RSS(self.y1, predictions3), TSS(self.y1))
        self.assertAlmostEqual(RSS(self.y1, predictions4), TSS(self.y1))
        self.assertAlmostEqual(RSS(self.y1, predictions5), TSS(self.y1))


if __name__ == "__main__":
    unittest.main()