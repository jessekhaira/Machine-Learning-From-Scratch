""" This module contains unit tests for a variety of
activation functions """
import numpy as np
import unittest
from machine_learning_algorithms.neural_net_utility.activation_functions import (
    Softmax, Sigmoid, TanH, ReLU, LeakyReLU, IdentityActivation)
from machine_learning_algorithms.neural_net_utility.loss_functions import CrossEntropy
from machine_learning_algorithms.neural_net_utility.neural_net_layers import dl_dz_softmax


class TestSoftmaxActivation(unittest.TestCase):
    """ This class contains tests for the softmax activation function"""

    @classmethod
    def setUpClass(cls):
        # these functions are completely stateless so we can initialize
        # them inside the class and reuse them among tests
        cls.softmax = Softmax()
        cls.cross_entropy = CrossEntropy()
        cls.rs = np.random.RandomState(32)

    def test_forward1(self):
        # logits represent 4 classes, 5 examples
        z = TestSoftmaxActivation.rs.randn(4, 5)
        a = TestSoftmaxActivation.softmax.compute_output(z)
        self.assertTrue(isinstance(a, np.ndarray))
        # sum of every examples activated values should sum up to 1
        # according to softmax
        self.assertTrue(np.all(np.isclose(np.sum(a, axis=0, keepdims=True), 1)))

    def test_forward2(self):
        z = TestSoftmaxActivation.rs.randn(55, 128)
        a = TestSoftmaxActivation.softmax.compute_output(z)
        self.assertTrue(np.all(np.isclose(np.sum(a, axis=0, keepdims=True), 1)))

    def test_backward1(self):
        x = TestGradientChecking.rs.randn(3, 1)
        output_arr = TestGradientChecking.softmax.gradient_checking(
            x, num_checks=5)
        self.assertTrue(np.all(output_arr <= 1e-10))

    def test_backward2(self):
        x = TestGradientChecking.rs.randn(30, 1)
        output_arr = TestGradientChecking.softmax.gradient_checking(
            x, num_checks=21)
        self.assertTrue(np.all(output_arr <= 1e-8))

    def test_backward3(self):
        x = TestGradientChecking.rs.randn(350, 1)
        output_arr = TestGradientChecking.softmax.gradient_checking(
            x, num_checks=50)
        self.assertTrue(np.all(output_arr <= 1e-7))


class TestSigmoid(unittest.TestCase):
    """ This class tests the forward and backward pass for the sigmoid
    activation function"""

    @classmethod
    def setUpClass(cls):
        # completely stateless so we can initialize
        # them inside the class and reuse them among tests
        cls.sigmoid = Sigmoid()
        cls.rs = np.random.RandomState(32)

    def test_forward1(self):
        x = TestSigmoid.rs.randn(1, 10)
        output = TestSigmoid.sigmoid.compute_output(x)
        self.assertTrue(np.all((output >= 0) & (output <= 1)))

    def test_forward2(self):
        x = TestSigmoid.rs.randn(5, 50)
        output = TestSigmoid.sigmoid.compute_output(x)
        self.assertTrue(np.all((output >= 0) & (output <= 1)))

    def test_forward3(self):
        x = TestSigmoid.rs.randn(1550, 1000)
        output = TestSigmoid.sigmoid.compute_output(x)
        self.assertTrue(np.all((output >= 0) & (output <= 1)))

    def test_backward1(self):
        x = TestGradientChecking.rs.randn(3, 1)
        output_arr = TestGradientChecking.sigmoid.gradient_checking(
            x, num_checks=5)
        self.assertTrue(np.all(output_arr <= 1e-10))

    def test_backward2(self):
        x = TestGradientChecking.rs.randn(30, 1)
        output_arr = TestGradientChecking.sigmoid.gradient_checking(
            x, num_checks=5)
        self.assertTrue(np.all(output_arr <= 1e-10))

    def test_backward3(self):
        x = TestGradientChecking.rs.randn(350, 1)
        output_arr = TestGradientChecking.sigmoid.gradient_checking(
            x, num_checks=5)
        self.assertTrue(np.all(output_arr <= 1e-10))


class TestLeakyReLU(unittest.TestCase):
    """ This class tests the Leaky ReLU activation function"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.leaky_relu = LeakyReLU()


class TestGradientChecking(unittest.TestCase):
    """ This class tests the gradient checking method for some activation
    functions"""

    @classmethod
    def setUpClass(cls):
        cls.softmax = Softmax()
        cls.rs = np.random.RandomState(40)
        cls.sigmoid = Sigmoid()
        cls.tanh = TanH()

    def test_tanh1(self):
        x = TestGradientChecking.rs.randn(3, 1)
        output_arr = TestGradientChecking.tanh.gradient_checking(x,
                                                                 num_checks=5)
        self.assertTrue(np.all(output_arr <= 1e-10))

    def test_tanh2(self):
        x = TestGradientChecking.rs.randn(30, 1)
        output_arr = TestGradientChecking.tanh.gradient_checking(x,
                                                                 num_checks=5)
        self.assertTrue(np.all(output_arr <= 1e-10))

    def test_tanh3(self):
        x = TestGradientChecking.rs.randn(350, 1)
        output_arr = TestGradientChecking.tanh.gradient_checking(x,
                                                                 num_checks=5)
        self.assertTrue(np.all(output_arr <= 1e-9))


if __name__ == "__main__":
    unittest.main()
