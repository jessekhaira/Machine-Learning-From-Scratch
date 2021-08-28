""" This module contains unit tests for a variety of
activation functions """
import numpy as np
import unittest
from machine_learning_algorithms.neural_net_utility.activation_functions import Softmax
from machine_learning_algorithms.neural_net_utility.loss_functions import cross_entropy
from machine_learning_algorithms.neural_net_utility.neural_net_layers import dl_dz_softmax


class TestActivationFunctions(unittest.TestCase):
    """ This class contain unit tests for activation functions """

    def test1(self):
        #test with single example first
        np.random.seed(21)
        softmax_predictor = Softmax()
        ce = cross_entropy()
        z = np.random.randn(4, 5)
        a = softmax_predictor.compute_output(z)
        print(a)
        y = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0],
                      [0, 1, 0, 0]]).T
        print(ce.get_loss(y, a))
        print(y)
        dl_da = ce.get_gradient_pred(y, a)
        print(dl_da)
        dl_dz = dl_dz_softmax(z, a, dl_da, softmax_predictor)
        # Only one value in each column should be negative,
        # the rest should be positive and the values should
        # be easily correlated with (a-1)/m (if y =1) else
        # (a)/m
        print(dl_dz)


if __name__ == "__main__":
    unittest.main()