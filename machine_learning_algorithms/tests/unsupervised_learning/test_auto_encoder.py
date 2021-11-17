""" This module contains unit tests for the auto encoder
algorithm """
import unittest
import numpy as np
import tensorflow as tf
from machine_learning_algorithms.unsupervised_learning.auto_encoder import AutoEncoder
from machine_learning_algorithms.unsupervised_learning.auto_encoder import GradientDescentMomentum


class AutoEncoderTests(unittest.TestCase):
    """ This class contains unit tests for the auto encoder
    algorithm """

    def setUp(self) -> None:
        (self.x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
        self.x_train = np.array(self.x_train, dtype=float)
        self.x_train /= 255
        self.x_train.reshape(784, -1)
        self.x_train = self.x_train.reshape(784, -1)
        return super().setUp()

    def test1(self):
        # works fine, but takes a really long time to train
        autoencoder_object = AutoEncoder(size_encoding=150)
        autoencoder_object.fit(self.x_train,
                               num_epochs=10000,
                               optim=GradientDescentMomentum(),
                               ret_train_loss=True,
                               verbose=True,
                               learn_rate=0.1,
                               batch_size=512)


if __name__ == "__main__":
    unittest.main()