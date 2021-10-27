""" This module contains unit tests for various convolutional
network layers """
import numpy as np
import unittest
from machine_learning_algorithms.neural_net_utility.convolutional_layers import Conv2D, Pool
from machine_learning_algorithms.neural_net_utility.activation_functions import ReLU, IdentityActivation


class TestConvolutionalLayers(unittest.TestCase):
    """ This class contains unit tests for variosu convolutional network
    layers """

    def test_padding(self):
        # 10 pictures that are 3 x 5 x 3
        noise_images = np.random.randn(10, 3, 5, 3)
        conv_layer = Conv2D(filter_size=3,
                            input_depth=3,
                            num_filters=3,
                            activation_function=ReLU(),
                            padding="same",
                            stride=1)
        # looks good
        conv_layer.compute_forward(noise_images)

    def test_conv(self):
        x_shape = (2, 3, 4, 4)
        w_shape = (3, 3, 4, 4)
        x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
        w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
        b = np.linspace(-0.1, 0.2, num=3)

        conv_layer = Conv2D(filter_size=4,
                            input_depth=3,
                            num_filters=3,
                            activation_function=IdentityActivation(),
                            padding="same",
                            stride=2)
        conv_layer.filters = w
        conv_layer.b = b
        conv_layer.compute_forward(x)
        correct_out = np.array([[[[-0.08759809, -0.10987781],
                                  [-0.18387192, -0.2109216]],
                                 [[0.21027089, 0.21661097],
                                  [0.22847626, 0.23004637]],
                                 [[0.50813986, 0.54309974],
                                  [0.64082444, 0.67101435]]],
                                [[[-0.98053589, -1.03143541],
                                  [-1.19128892, -1.24695841]],
                                 [[0.69108355, 0.66880383],
                                  [0.59480972, 0.56776003]],
                                 [[2.36270298, 2.36904306],
                                  [2.38090835, 2.38247847]]]])

        xd = rel_error(conv_layer.Z, correct_out)
        print(xd)
        # manually replacing some of the parameters w/ the given in the conv
        # object leads to the correct ouput

    def test_pool(self):
        x_shape = (2, 3, 4, 4)
        x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)

        pool_layer = Pool(filter_size=2, stride=2, poolType="max")
        pool_layer.compute_forward(x)

        correct_out = np.array([[[[-0.26315789, -0.24842105],
                                  [-0.20421053, -0.18947368]],
                                 [[-0.14526316, -0.13052632],
                                  [-0.08631579, -0.07157895]],
                                 [[-0.02736842, -0.01263158],
                                  [0.03157895, 0.04631579]]],
                                [[[0.09052632, 0.10526316],
                                  [0.14947368, 0.16421053]],
                                 [[0.20842105, 0.22315789],
                                  [0.26736842, 0.28210526]],
                                 [[0.32631579, 0.34105263], [0.38526316,
                                                             0.4]]]])

        xd = rel_error(pool_layer.Z, correct_out)
        print(xd)


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


if __name__ == "__main__":
    unittest.main()
