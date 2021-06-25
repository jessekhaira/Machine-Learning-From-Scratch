from machine_learning_algorithms.unsupervised_learning.auto_encoder import Deep_Autoencoder
from machine_learning_algorithms.unsupervised_learning.auto_encoder import Adam, gradientDescentMomentum
import unittest
import numpy as np
import tensorflow as tf

(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = np.array(x_train, dtype=float)
x_train /= 255
x_train.reshape(784, -1)
x_train = x_train.reshape(784, -1)


class tests(unittest.TestCase):

    def test1(self):
        # works fine, but takes a really long time to train
        autoencoderObj = Deep_Autoencoder(size_encoding=150)
        autoencoderObj.fit(x_train,
                           num_epochs=10000,
                           optim=gradientDescentMomentum(),
                           ret_train_loss=True,
                           verbose=True,
                           learn_rate=0.1,
                           batch_size=512)


if __name__ == "__main__":
    unittest.main()