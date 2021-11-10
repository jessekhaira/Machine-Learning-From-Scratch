""" This module contains code for testing the ensemble tree models
for the supervised machine learning task of regression """
import unittest
from sklearn.datasets import load_boston
from machine_learning_algorithms.supervised_learning.regression.bagged_forest_regressor import BaggedForestRegression
from machine_learning_algorithms.supervised_learning.regression.random_forest_regressor import RandomForestRegressor


class TestEnsembleTreesRegression(unittest.TestCase):
    """ This class contains unit tests for ensemble tree models 
    performing regression """

    def setUp(self) -> None:
        self.x, self.y = load_boston(return_X_y=True)
        self.x = self.x.T
        self.y = self.y.T.reshape(1, -1)
        return super().setUp()

    def testRF_Regressor(self):
        # Use 5 random bootstrapped samples to train each tree and then get OOB mse and rmse
        # should be quite low
        mod2 = RandomForestRegressor(verbose=True,
                                     bootstrap=True,
                                     max_samples=5,
                                     minSamplesSplit=1,
                                     maxFeatures=int(self.x.shape[0]**0.5))
        mod2.fit(self.x, self.y)
        mse2, rmse2 = mod2.get_oob_score(self.x, self.y)
        print(mse2, rmse2)
        self.assertGreaterEqual(mse2, 20)
        self.assertLessEqual(rmse2, 10)

        # fit on a slightly bigger portion and see what happens - should be better than before
        mod4 = RandomForestRegressor(verbose=True,
                                     bootstrap=True,
                                     max_samples=15,
                                     minSamplesSplit=3,
                                     maxFeatures=int(self.x.shape[0]**0.5))
        mod4.fit(self.x, self.y)
        mse4, rmse4 = mod4.get_oob_score(self.x, self.y)
        print(mse4, rmse4)
        self.assertLessEqual(mse4, mse2)
        self.assertLessEqual(rmse4, rmse2)

        # allow each tree to see a large bootstrapped sample and get mse and rmse - should perform quite well
        # may not perform as well as just one decision tree on this particular dataset since the training procedure has so much stochasiticy in it
        # and it is already difficult to overfit to this dataset
        mod5 = RandomForestRegressor(verbose=True,
                                     bootstrap=True,
                                     max_samples=300,
                                     minSamplesSplit=25,
                                     maxFeatures=int(self.x.shape[0]**0.5),
                                     maxDepth=4,
                                     min_impurity_decrease=0.25)
        mod5.fit(self.x, self.y)
        mse5, rmse5 = mod5.get_oob_score(self.x, self.y)
        print(mse5, rmse5)
        self.assertLessEqual(mse5, mse4)
        self.assertLessEqual(rmse5, rmse4)

    def testBaggedForestRegressor(self):
        ## Diff b/w bagged forest and random forest - bagged forests sees every single feature at every single split point
        # can achieve lower bias since there is naturally less stochasiticy in the training procedure, but will struggle with overfitting
        # and lack of variety in the ensemble
        mod2 = BaggedForestRegression(verbose=True,
                                      bootstrap=True,
                                      max_samples=5,
                                      minSamplesSplit=1)
        mod2.fit(self.x, self.y)
        mse2, rmse2 = mod2.get_oob_score(self.x, self.y)
        print(mse2, rmse2)
        self.assertGreaterEqual(mse2, 0.90)
        self.assertGreaterEqual(rmse2, 0.05)

        # fit on a slightly bigger portion and see what happens - should be better than before
        mod4 = BaggedForestRegression(verbose=True,
                                      bootstrap=True,
                                      max_samples=15,
                                      minSamplesSplit=3)
        mod4.fit(self.x, self.y)
        mse4, rmse4 = mod4.get_oob_score(self.x, self.y)
        print(mse4, rmse4)
        self.assertLessEqual(mse4, mse2)
        self.assertLessEqual(rmse4, rmse2)

        # allow each tree to see a large bootstrapped sample and get mse and rmse - should performed quite well
        # may not perform as well as just one decision tree on this particular dataset since the training procedure has so much stochasicity in it
        # and it is already difficult to overfit to this dataset
        mod5 = BaggedForestRegression(verbose=True,
                                      bootstrap=True,
                                      max_samples=150,
                                      minSamplesSplit=20)
        mod5.fit(self.x, self.y)
        mse5, rmse5 = mod5.get_oob_score(self.x, self.y)
        print(mse5, rmse5)
        self.assertLessEqual(mse5, mse4)
        self.assertLessEqual(rmse5, rmse4)


if __name__ == "__main__":
    unittest.main()