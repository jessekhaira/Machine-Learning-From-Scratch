""" This module contains code related to testing the k nearest neighbours
classifier """
from machine_learning_algorithms.supervised_learning.classifiers.k_nearest_neighbours_classifier import KNearestNeighboursClassifier
import unittest
import numpy as np
import tensorflow as tf
from machine_learning_algorithms.utility.ScoreFunctions import accuracy

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


def reshapeX(x_train, x_test):
    return x_train.reshape(x_train.shape[0],
                           -1), x_test.reshape(x_test.shape[0], -1)


x1_train, x1_test = reshapeX(x_train, x_test)


class KNNClassifierTests(unittest.TestCase):
    """
    This class contains unit tests for each of the methods
    defined for the kNN class.
    """

    def testAttributes(self):
        obj1 = KNearestNeighboursClassifier()
        self.assertEqual(obj1.k, 10)
        self.assertEqual(obj1.similarity_metric, "L2")

    def test_xy_fit(self):
        #generate a np matrix for x_train and y_train and test if it fits properly
        x_train = np.random.randn(10, 20)
        y_train = np.random.randn(10, 1)
        obj2 = KNearestNeighboursClassifier()
        obj2.fit(x_train, y_train)
        self.assertEqual(np.sum(obj2.model_x - x_train), 0)
        self.assertEqual(np.sum(obj2.model_y - y_train), 0)

    def test_overall_model(self):
        #object should be able to get >= 20% percent accuracy on CIFAR-10 with k = 10
        test_obj = KNearestNeighboursClassifier()
        test_obj.fit(x1_train, y_train)
        prediction_test = test_obj.predict(x1_test[:200])
        acc = accuracy(y_test, prediction_test)
        self.assertEqual(prediction_test.shape, y_test[:200].shape)
        self.assertGreaterEqual(acc, 20)


if __name__ == "__main__":
    unittest.main()