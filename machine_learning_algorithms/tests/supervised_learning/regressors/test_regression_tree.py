""" This module contains tests for the regression decision tree
algorithm """
import unittest
from sklearn.datasets import load_boston
from machine_learning_algorithms.supervised_learning.regression.regression_tree import RegressionTree
from machine_learning_algorithms.utility.score_functions import root_mean_squared_error, mean_squared_error, mean_absolute_error, total_sum_of_squares, residual_sum_of_squares
from machine_learning_algorithms.utility.k_fold_cross_validation import KFoldCrossValidation


class RegressionTreeTests(unittest.TestCase):
    """ This class contains test for the regression decision tree
    algorithm """

    def setUp(self):
        self.x1, self.y1 = load_boston(return_X_y=True)
        self.x1 = self.x1.T
        self.y1 = self.y1.T.reshape(1, -1)
        self.k_cv = KFoldCrossValidation()

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
        rmse = root_mean_squared_error(self.y1, predictions)
        mae = mean_absolute_error(self.y1, predictions)
        mse = mean_squared_error(self.y1, predictions)
        rmse = root_mean_squared_error(self.y1, predictions)
        self.assertEqual(mae, 0)
        self.assertEqual(mse, 0)
        self.assertEqual(rmse, 0)

        # test generalization of regression tree
        regression_obj2 = RegressionTree(minSamplesSplit=1)
        kScoreRMSE = self.k_cv.get_k_score(self.x1, self.y1,
                                           root_mean_squared_error,
                                           regression_obj2)
        kScoreMSE = self.k_cv.get_k_score(self.x1, self.y1, mean_squared_error,
                                          regression_obj2)
        kScoreMAE = self.k_cv.get_k_score(self.x1, self.y1, mean_absolute_error,
                                          regression_obj2)
        # Dataset is easy so we should expect 0 error
        self.assertEqual(kScoreRMSE, 0)
        self.assertEqual(kScoreMSE, 0)
        self.assertEqual(kScoreMAE, 0)

    def test_sanityChecks(self):
        # If we regularize the tree, we should get a higher
        # error than if we don't
        regression_obj2 = RegressionTree(minSamplesSplit=5,
                                         maxDepth=3,
                                         min_impurity_decrease=0.15)
        regression_obj2.fit(self.x1, self.y1)
        predictions2 = regression_obj2.predict(self.x1)
        error2 = root_mean_squared_error(self.y1, predictions2)

        # Sanity checks - regularization is so high all we get is one leaf
        # meaning all predictions are equal to mean of labels
        # meaning that the RSS of the predictions should be equal to
        # TSS of the label
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
        self.assertAlmostEqual(residual_sum_of_squares(self.y1, predictions3),
                               total_sum_of_squares(self.y1))
        self.assertAlmostEqual(residual_sum_of_squares(self.y1, predictions4),
                               total_sum_of_squares(self.y1))
        self.assertAlmostEqual(residual_sum_of_squares(self.y1, predictions5),
                               total_sum_of_squares(self.y1))


if __name__ == "__main__":
    unittest.main()