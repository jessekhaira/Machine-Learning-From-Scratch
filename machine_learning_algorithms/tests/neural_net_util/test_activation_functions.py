""" This module contains unit tests for a variety of
activation functions """
import numpy as np
import unittest
from machine_learning_algorithms.neural_net_utility.activation_functions import Softmax
from machine_learning_algorithms.neural_net_utility.loss_functions import CrossEntropy
from machine_learning_algorithms.neural_net_utility.neural_net_layers import dl_dz_softmax


class TestSoftmaxActivation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # these functions are completely stateless so we can initialize
        # them inside the class and reuse them among tests
        cls.softmax = Softmax()
        cls.cross_entropy = CrossEntropy()
        cls.rs = np.random.RandomState(32)

    def test_softmax1(self):
        # logits represent 4 classes, 5 examples
        z = TestSoftmaxActivation.rs.randn(4, 5)
        a = TestSoftmaxActivation.softmax.compute_output(z)
        self.assertTrue(isinstance(a, np.ndarray))
        # every example should sum up to 1
        self.assertTrue(np.all(np.isclose(np.sum(a, axis=0, keepdims=True), 1)))

    def test_softmax_backward1(self):
        a = np.array([[0.2, 0.8, 0, 0]]).reshape(4, 1)
        # definition for da/dz for softmax, ie: when i = j
        # and when i != j captured by the below
        expected = np.diagflat(a) - a.dot(a.T)
        self.assertTrue(
            np.all(expected ==
                   TestSoftmaxActivation.softmax.get_derivative_wrt_input(a)))

    def test_softmax_backward2(self):
        a = np.array([[0.5, 0, 0.5]]).reshape(3, 1)
        # definition for da/dz for softmax
        expected = np.array([[0.25, 0, -0.25], [0, 0, 0], [-0.25, 0, 0.25]])
        self.assertTrue(
            np.all(expected ==
                   TestSoftmaxActivation.softmax.get_derivative_wrt_input(a)))

    def test_softmax_backward3(self):
        a = np.array([0.05 for i in range(20)]).reshape(20, 1)
        expected = np.diagflat(a) - a.dot(a.T)
        self.assertTrue(
            np.all(expected ==
                   TestSoftmaxActivation.softmax.get_derivative_wrt_input(a)))


if __name__ == "__main__":
    unittest.main()
