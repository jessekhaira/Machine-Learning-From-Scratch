""" This module contains tests for objective functions """
import numpy as np
import unittest
from machine_learning_algorithms.neural_net_utility.loss_functions import CrossEntropy


class TestObjectiveFunctions(unittest.TestCase):
    """ This class contains unit tests for a variety of objective
    functions """

    def setUp(self):
        self.cross_entropy_object = CrossEntropy()

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


if __name__ == "__main__":
    unittest.main()
