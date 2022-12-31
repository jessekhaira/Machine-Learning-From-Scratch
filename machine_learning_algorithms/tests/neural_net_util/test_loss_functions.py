""" This module contains tests for objective functions """
import numpy as np
import unittest
from machine_learning_algorithms.neural_net_utility.loss_functions import CrossEntropy


class TestCrossEntropy(unittest.TestCase):
    """ This class contains unit tests for the cross entropy
    function """

    @classmethod
    def setUpClass(cls):
        cls.cross_entropy_object = CrossEntropy()
        cls.rs = np.random.RandomState(32)

    def test_cross_entropy_loss1(self):
        y = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]).T
        yhat = np.array([[0.2, 0.4, 0.2, 0.2], [0.5, 0.2, 0.2, 0.1],
                         [0.4, 0.1, 0.2, 0.9]]).T

        self.assertAlmostEqual(self.cross_entropy_object.get_loss(y, yhat),
                               0.5715994760306422)

    def test_cross_entropy_gradient1(self):
        y = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]).T
        yhat = np.array([[0.2, 0.4, 0.2, 0.2], [0.5, 0.2, 0.2, 0.1],
                         [0.4, 0.1, 0.2, 0.9]]).T
        jacobian_matrix = np.array([[0, -0.66666667, 0], [-0.83333333, 0, 0],
                                    [0, 0, 0], [0, 0, -0.37037037]])

        self.assertEqual(
            np.allclose(self.cross_entropy_object.get_gradient_pred(y, yhat),
                        jacobian_matrix), True)

    def test_cross_entropy_loss_2(self):
        y1 = np.array([0, 0, 1, 0]).T
        yhat1 = np.array([0.32574286, 0.081362, 0.0352241, 0.55767104]).T
        self.assertAlmostEqual(self.cross_entropy_object.get_loss(y1, yhat1),
                               3.346024771559287)

    def test_cross_entropy_gradient2(self):
        y = np.array([0, 0, 1, 0]).T
        yhat = np.array([0.32574286, 0.081362, 0.0352241, 0.55767104]).T
        jacobian_matrix = np.array([[0, 0, -28.38965367, 0]]).T
        gradient = self.cross_entropy_object.get_gradient_pred(
            y.reshape(-1, 1), yhat.reshape(-1, 1))
        self.assertAlmostEqual(np.allclose(gradient, jacobian_matrix), True)

    def test_backward1(self):
        # shape (C, m) where m is number of examples == 1 in this case
        # with 4 classes
        y = np.array([1500, 2500, 49000, 5012]).reshape(-1, 1)
        yhat = np.array([2, 5, 1, 9], dtype=np.float64).reshape(-1, 1)
        rel_error_grad_array, rel_error_computed_vectors = (
            TestCrossEntropy.cross_entropy_object.gradient_checking(
                y, yhat, num_checks=2))

        print(rel_error_computed_vectors)
        self.assertTrue(np.all(rel_error_grad_array <= 1e-8))
        self.assertTrue(np.all(rel_error_computed_vectors <= 1e-8))


if __name__ == "__main__":
    unittest.main()
