import tensorflow as tf
import numpy as np
from machine_learning_algorithms.unsupervised_learning.principal_component_analysis import PrincipalComponentAnalysis
import unittest
import matplotlib.pyplot as plt

(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = np.array(x_train, dtype=float)
x_train /= 255

x_trainOwn = x_train.reshape(-1, 60000)


class tests(unittest.TestCase):

    def test1(self):
        # looks fine
        pca = PrincipalComponentAnalysis()
        transformed_X = pca.fit_transform(x_trainOwn, 10)
        plt.scatter(transformed_X[0, :200], transformed_X[1, :200])


if __name__ == "__main__":
    unittest.main()