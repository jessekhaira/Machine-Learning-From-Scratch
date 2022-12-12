""" This module contains unit tests for a variety of
activation functions """
import numpy as np
import unittest
from machine_learning_algorithms.neural_net_utility.activation_functions import Softmax
from machine_learning_algorithms.neural_net_utility.loss_functions import CrossEntropy
from machine_learning_algorithms.neural_net_utility.neural_net_layers import dl_dz_softmax


class TestActivationFunctions(unittest.TestCase):
    """ This class contain unit tests for activation functions """

    @classmethod
    def setUpClass(cls):
        # these functions are completely stateless so we can initialize
        # them inside the class and reuse them among tests
        cls.softmax = Softmax()
        cls.cross_entropy = CrossEntropy()
        cls.rs = np.random.RandomState(32)

    def test_softmax1(self):
        # logits represent 4 classes, 5 examples
        z = TestActivationFunctions.rs.randn(4, 5)
        a = TestActivationFunctions.softmax.compute_output(z)
        self.assertTrue(isinstance(a, np.ndarray))
        # every example should sum up to 1
        self.assertTrue(np.all(np.isclose(np.sum(a, axis=0, keepdims=True), 1)))


if __name__ == "__main__":
    unittest.main()
