""" This module represents a unit test created for restricted boltzmann
machines """
import tensorflow as tf
import numpy as np
from ML_algorithms.Unsupervised_Learning.RestrictedBoltzmannMachine import RBM
import unittest
import matplotlib.pyplot as plt

(x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
x_train = np.array(x_train, dtype=float)

# dividing by 255 is a form of min-max normalization -> normalizes every
# value in between 0 and 1
x_train /= 255
# reshape to (m,n) -> RBMS accept batches of vectors w/ n_vis features, not
# images, so we flatten each image to a vector and feed it into the RBM
x_train = x_train.reshape(50000, 3072)


class RBMTest(unittest.TestCase):

    def test1(self):
        # reconstruction error should fall rapidly and consistently at the start
        # of learning and then more slowly
        rbm = RBM(n_visible=x_train.shape[1],
                  n_hidden=256,
                  is_img=True,
                  k=5,
                  img_d=3,
                  img_h=32,
                  img_w=32,
                  batch_size=10,
                  n_epochs=2000,
                  learning_rate=0.1,
                  weight_decay=0.001)
        rbm.train(x_train)

    def test_reconstruct(self):
        np.random.seed(71)
        rbm = RBM(n_visible=x_train.shape[1])
        output = rbm.reconstruct(x_train)
        random_indices = np.random.randint(0, output.shape[0], 10)

        _, ax = plt.subplots(nrows=2, ncols=5)
        i = 0
        for ri, row in enumerate(ax):
            for col in row:
                if ri == 0:
                    col.imshow(x_train[random_indices[i], :].reshape(32, 32, 3))
                else:
                    col.imshow(output[random_indices[i], :].reshape(32, 32, 3))
                i += 1
        plt.show()


if __name__ == "__main__":
    unittest.main()
