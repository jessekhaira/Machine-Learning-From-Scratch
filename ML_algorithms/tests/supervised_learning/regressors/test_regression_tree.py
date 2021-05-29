""" This module contains tests for the regression decision tree
algorithm """
import unittest
from sklearn.datasets import load_boston
from ML_algorithms.Supervised_Learning.Regression.regression_tree import RegressionTree
from ML_algorithms.Utility.ScoreFunctions import RMSE, MSE, MAE, TSS, RSS
from ML_algorithms.Utility.k_Fold_CV import k_fold_CV

X1, Y1 = load_boston(return_X_y=True)
X1 = X1.T
Y1 = Y1.T.reshape(1, -1)
kCV = k_fold_CV()


class RegressionTreeTests(unittest.TestCase):
    """ This class contains test for the regression decision tree
    algorithm """

    def testFit(self):
        # The tree is not being constrained when it is fitting
        # thus we should be able to achieve 0 error AKA all residuals are 0
        regressionObj = RegressionTree(minSamplesSplit=1)
        regressionObj.fit(X1, Y1)
        predictions = regressionObj.predict(X1)
        rmse = RMSE(Y1, predictions)
        mae = MAE(Y1, predictions)
        mse = MSE(Y1, predictions)
        rmse = RMSE(Y1, predictions)
        self.assertEqual(mae, 0)
        self.assertEqual(mse, 0)
        self.assertEqual(rmse, 0)

        # test generalization of regression tree
        regressionObj2 = RegressionTree(minSamplesSplit=1)
        kScoreRMSE = kCV.getKScore(X1, Y1, RMSE, regressionObj2)
        kScoreMSE = kCV.getKScore(X1, Y1, MSE, regressionObj2)
        kScoreMAE = kCV.getKScore(X1, Y1, MAE, regressionObj2)
        # Dataset is easy so we should expect 0 error
        self.assertEqual(kScoreRMSE, 0)
        self.assertEqual(kScoreMSE, 0)
        self.assertEqual(kScoreMAE, 0)

    def test_sanityChecks(self):
        # If we regularize the tree, we should get a higher error than if we don't
        regressionObj2 = RegressionTree(minSamplesSplit=5,
                                        maxDepth=3,
                                        min_impurity_decrease=0.15)
        regressionObj2.fit(X1, Y1)
        predictions2 = regressionObj2.predict(X1)
        error2 = RMSE(Y1, predictions2)

        # Sanity checks - regularization is so high all we get is one leaf
        # meaning all predictions are equal to mean of labels
        # meaning that the RSS of the predictions should be equal to TSS of the label
        regressionObj3 = RegressionTree(minSamplesSplit=10,
                                        maxDepth=0,
                                        min_impurity_decrease=0.15)
        regressionObj3.fit(X1, Y1)
        predictions3 = regressionObj3.predict(X1)
        error3 = RMSE(Y1, predictions3)

        regressionObj4 = RegressionTree(minSamplesSplit=1000,
                                        maxDepth=10,
                                        min_impurity_decrease=0.15)
        regressionObj4.fit(X1, Y1)
        predictions4 = regressionObj4.predict(X1)
        error4 = RMSE(Y1, predictions4)

        regressionObj5 = RegressionTree(minSamplesSplit=10,
                                        maxDepth=10,
                                        min_impurity_decrease=1)
        regressionObj5.fit(X1, Y1)
        predictions5 = regressionObj4.predict(X1)
        error5 = RMSE(Y1, predictions5)

        self.assertGreaterEqual(error2, 0)
        self.assertAlmostEqual(RSS(Y1, predictions3), TSS(Y1))
        self.assertAlmostEqual(RSS(Y1, predictions4), TSS(Y1))
        self.assertAlmostEqual(RSS(Y1, predictions5), TSS(Y1))


if __name__ == "__main__":
    unittest.main()