""" This module contains unit tests for the pca algorithm """
import tensorflow as tf
import numpy as np
from machine_learning_algorithms.unsupervised_learning.principal_component_analysis import PrincipalComponentAnalysis
import unittest
import matplotlib.pyplot as plt


class TestPCA(unittest.TestCase):
    """ This class contains unit tests for the pca
    algorithm """

    def setUp(self) -> None:
        (x, _), (_, _) = tf.keras.datasets.mnist.load_data()
        x = np.array(x, dtype=float)
        x /= 255

        self.x_train = x.reshape(-1, 60000)
        return super().setUp()

    def test1(self):
        pca = PrincipalComponentAnalysis()
        transformed_x = pca.fit_transform(self.x_train, 10)
        plt.scatter(transformed_x[0, :200], transformed_x[1, :200])


if __name__ == "__main__":
    unittest.main()
