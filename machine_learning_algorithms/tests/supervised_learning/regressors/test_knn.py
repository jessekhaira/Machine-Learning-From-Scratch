""" This module contains unit tests for the k nearest neighbour
algorithm for regression """
import unittest
import numpy as np
import sklearn.metrics
import sklearn.model_selection
import sklearn.datasets
from machine_learning_algorithms.supervised_learning.regression.k_nearest_neighbours_regressor import KNearestNeighboursRegressor
from machine_learning_algorithms.utility.score_functions import mean_squared_error, mean_absolute_error, root_mean_squared_error


class KNearestNeighboursRegressorTests(unittest.TestCase):
    """ This class contains unit tests for the k nearest
    neighbours algorithm for regression """

    def test_xy_fit(self):
        """ Generate a numpy matrix for x_train and y_train and test
        if it fits properly """
        x_train = np.random.randn(10, 20)
        y_train = np.random.randn(10, 1)
        obj2 = KNearestNeighboursRegressor()
        obj2.fit(x_train, y_train)
        self.assertEqual(np.sum(obj2.model_x - x_train), 0)
        self.assertEqual(np.sum(obj2.model_y - y_train), 0)

    def test_overall_model(self):
        """ This implementation of k nearest neighbours should be
        able to achieve mean_absolute_error <=10, mean_squared_error <= 50 and root_mean_squared_error <=10 if everything
        is wired correctly """
        x, y = sklearn.datasets.load_boston(return_X_y=True)
        xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(
            x, y)
        ytest = ytest.reshape(-1, 1)
        test_obj = KNearestNeighboursRegressor()
        test_obj.fit(xtrain, ytrain)
        prediction_test = test_obj.predict(xtest)
        self.assertEqual(prediction_test.shape, ytest.shape)
        mean_abs = mean_absolute_error(ytest, prediction_test)
        mean_sq = mean_squared_error(ytest, prediction_test)
        root_meansq = root_mean_squared_error(ytest, prediction_test)
        self.assertLessEqual(mean_abs, 10)
        self.assertLessEqual(mean_sq, 65)
        self.assertLessEqual(root_meansq, 10)


if __name__ == "__main__":
    unittest.main()
