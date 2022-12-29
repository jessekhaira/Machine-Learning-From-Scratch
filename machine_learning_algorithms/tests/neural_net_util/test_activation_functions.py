""" This module contains unit tests for a variety of
activation functions """
import numpy as np
import unittest
from typing import cast
from machine_learning_algorithms.neural_net_utility.activation_functions import (
    Softmax, Sigmoid, TanH, ReLU, LeakyReLU, IdentityActivation)
from machine_learning_algorithms.neural_net_utility.loss_functions import CrossEntropy
from machine_learning_algorithms.neural_net_utility.neural_net_layers import dl_dz_softmax


class TestSoftmax(unittest.TestCase):
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
        z = TestSoftmax.rs.randn(4, 5)
        a = TestSoftmax.softmax.compute_output(z)
        self.assertTrue(isinstance(a, np.ndarray))
        # sum of every examples activated values should sum up to 1
        # according to softmax
        self.assertTrue(np.all(np.isclose(np.sum(a, axis=0, keepdims=True), 1)))

    def test_forward2(self):
        z = TestSoftmax.rs.randn(55, 128)
        a = TestSoftmax.softmax.compute_output(z)
        self.assertTrue(np.all(np.isclose(np.sum(a, axis=0, keepdims=True), 1)))

    def test_backward1(self):
        x = TestSoftmax.rs.randn(3, 1)
        output_arr = TestSoftmax.softmax.gradient_checking(x, num_checks=5)
        self.assertTrue(np.all(output_arr <= 1e-10))

    def test_backward2(self):
        x = TestSoftmax.rs.randn(30, 1)
        output_arr = TestSoftmax.softmax.gradient_checking(x, num_checks=21)
        self.assertTrue(np.all(output_arr <= 1e-8))

    def test_backward3(self):
        x = TestSoftmax.rs.randn(350, 1)
        output_arr = TestSoftmax.softmax.gradient_checking(x, num_checks=50)
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
        x = TestSigmoid.rs.randn(3, 1)
        output_arr = TestSigmoid.sigmoid.gradient_checking(x, num_checks=5)
        self.assertTrue(np.all(output_arr <= 1e-10))

    def test_backward2(self):
        x = TestSigmoid.rs.randn(30, 1)
        output_arr = TestSigmoid.sigmoid.gradient_checking(x, num_checks=5)
        self.assertTrue(np.all(output_arr <= 1e-10))

    def test_backward3(self):
        x = TestSigmoid.rs.randn(350, 1)
        output_arr = TestSigmoid.sigmoid.gradient_checking(x, num_checks=5)
        self.assertTrue(np.all(output_arr <= 1e-10))


class TestLeakyReLU(unittest.TestCase):
    """ This class tests the Leaky ReLU activation function"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.leaky_relu = LeakyReLU()
        cls.rs = np.random.RandomState(32)

    def test_forward1(self):
        x = TestLeakyReLU.rs.randn(1, 10)
        output = cast(np.ndarray, TestLeakyReLU.leaky_relu.compute_output(x))
        self.assertTrue(np.all((output[x > 0] - x[x > 0]) == 0))
        self.assertTrue(np.all((output[x < 0] - x[x < 0] * 0.01) == 0))

    def test_forward2(self):
        x = 5
        output = TestLeakyReLU.leaky_relu.compute_output(x)
        self.assertEqual(output, 5)

    def test_forward3(self):
        x = 5.1231
        output = TestLeakyReLU.leaky_relu.compute_output(x)
        self.assertEqual(output, 5.1231)

    def test_forward4(self):
        x = TestLeakyReLU.rs.randn(500, 1000)
        output = cast(np.ndarray, TestLeakyReLU.leaky_relu.compute_output(x))
        self.assertTrue(np.all((output[x > 0] - x[x > 0]) == 0))
        self.assertTrue(np.all((output[x < 0] - x[x < 0] * 0.01) == 0))

    def test_backward1(self):
        x = TestLeakyReLU.rs.randn(3, 1)
        output_arr = TestLeakyReLU.leaky_relu.gradient_checking(x, num_checks=5)
        self.assertTrue(np.all(output_arr <= 1e-11))

    def test_backward2(self):
        x = TestLeakyReLU.rs.randn(1, 1)
        output_arr = TestLeakyReLU.leaky_relu.gradient_checking(x,
                                                                num_checks=15)
        self.assertTrue(np.all(output_arr <= 1e-11))

    def test_backward3(self):
        x = 5
        output = TestLeakyReLU.leaky_relu.gradient_checking(x, num_checks=15)
        self.assertTrue(np.all(output <= 1e-10))

    def test_backward4(self):
        x = 5.123
        output = TestLeakyReLU.leaky_relu.gradient_checking(x, num_checks=25)
        self.assertTrue(np.all(output <= 1e-10))

    def test_backward5(self):
        x = TestLeakyReLU.rs.randn(100, 1000)
        output = TestLeakyReLU.leaky_relu.gradient_checking(x, num_checks=25)
        self.assertTrue(np.all(output <= 1e-9))


class TestTanH(unittest.TestCase):
    """ This class tests the TanH activation function"""

    @classmethod
    def setUpClass(cls):
        cls.rs = np.random.RandomState(40)
        cls.tanh = TanH()

    def test_forward1(self):
        x = TestTanH.rs.randn(1, 10)
        output = cast(np.ndarray, TestTanH.tanh.compute_output(x))
        self.assertTrue(np.all((output >= -1) & (output <= 1)))

    def test_forward2(self):
        x = 5
        output = TestTanH.tanh.compute_output(x)
        self.assertTrue((output - 1.0000908) <= 1e-8)

    def test_forward3(self):
        x = TestTanH.rs.randn(5, 50)
        output = cast(np.ndarray, TestTanH.tanh.compute_output(x))
        self.assertTrue(np.all((output >= -1) & (output <= 1)))

    def test_backward1(self):
        x = TestTanH.rs.randn(3, 1)
        output_arr = TestTanH.tanh.gradient_checking(x, num_checks=5)
        self.assertTrue(np.all(output_arr <= 1e-10))

    def test_backward2(self):
        x = TestTanH.rs.randn(30, 1)
        output_arr = TestTanH.tanh.gradient_checking(x, num_checks=5)
        self.assertTrue(np.all(output_arr <= 1e-10))

    def test_backward3(self):
        x = TestTanH.rs.randn(350, 1)
        output_arr = TestTanH.tanh.gradient_checking(x, num_checks=5)
        self.assertTrue(np.all(output_arr <= 1e-9))

    def test_backward4(self):
        x = TestTanH.rs.randn(1, 1)
        output_arr = TestTanH.tanh.gradient_checking(x, num_checks=50)
        self.assertTrue(np.all(output_arr <= 1e-10))


class TestReLU(unittest.TestCase):
    """ This class tests the ReLU activation function"""

    @classmethod
    def setUpClass(cls):
        cls.rs = np.random.RandomState(40)
        cls.ReLU = ReLU()

    def test_forward1(self):
        x = TestReLU.rs.randn(1, 10)
        output = cast(np.ndarray, TestReLU.ReLU.compute_output(x))
        self.assertTrue(np.all((output[x > 0] - x[x > 0]) == 0))
        self.assertTrue(np.all((output[x < 0]) == 0))

    def test_forward2(self):
        x = TestReLU.rs.randn(5, 50)
        output = cast(np.ndarray, TestReLU.ReLU.compute_output(x))
        self.assertTrue(np.all((output[x > 0] - x[x > 0]) == 0))
        self.assertTrue(np.all((output[x < 0]) == 0))

    def test_forward3(self):
        x = 12
        output = TestReLU.ReLU.compute_output(x)
        self.assertTrue(output, 12)

    def test_forward4(self):
        x = -15213.5123
        output = TestReLU.ReLU.compute_output(x)
        self.assertEqual(output, 0)

    def test_forward5(self):
        x = TestReLU.rs.randn(557, 600)
        output = TestReLU.ReLU.compute_output(x)
        self.assertTrue(np.all((output[x > 0] - x[x > 0]) == 0))
        self.assertTrue(np.all((output[x < 0]) == 0))

    def test_backward1(self):
        x = TestReLU.rs.randn(3, 1)
        output_arr = TestReLU.ReLU.gradient_checking(x, num_checks=10)
        self.assertTrue(np.all(output_arr <= 1e-10))

    def test_backward2(self):
        x = TestReLU.rs.randn(1, 1)
        output_arr = TestReLU.ReLU.gradient_checking(x, num_checks=15)
        self.assertTrue(np.all(output_arr <= 1e-11))

    def test_backward3(self):
        x = 5
        output = TestReLU.ReLU.gradient_checking(x, num_checks=15)
        self.assertTrue(np.all(output <= 1e-10))

    def test_backward4(self):
        x = 5.123
        output = TestReLU.ReLU.gradient_checking(x, num_checks=25)
        self.assertTrue(np.all(output <= 1e-10))

    def test_backward5(self):
        x = TestReLU.rs.randn(50, 70)
        output = TestReLU.ReLU.gradient_checking(x, num_checks=25)
        self.assertTrue(np.all(output <= 1e-9))


class TestIdentity(unittest.TestCase):
    """ This class tests the Identity activation function"""

    @classmethod
    def setUpClass(cls):
        cls.rs = np.random.RandomState(40)
        cls.identity = IdentityActivation()

    def test_forward1(self):
        x = TestIdentity.rs.randn(1, 10)
        output = cast(np.ndarray, TestIdentity.identity.compute_output(x))
        self.assertTrue(np.all(x == output))


if __name__ == "__main__":
    unittest.main()
