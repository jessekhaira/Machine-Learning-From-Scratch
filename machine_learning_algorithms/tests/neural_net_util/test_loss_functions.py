""" This module contains tests for objective functions """
import numpy as np
import unittest
from machine_learning_algorithms.neural_net_utility.loss_functions import CrossEntropy, NegativeLogLoss
from machine_learning_algorithms.utility.misc import rel_error, get_unique_counts_array, get_percent
from typing import cast


class TestCrossEntropy(unittest.TestCase):
    """ This class contains unit tests for the cross entropy
    function """

    @classmethod
    def setUpClass(cls):
        cls.cross_entropy_object = CrossEntropy()
        cls.rs = np.random.RandomState(32)

    def test_forward1(self):
        y = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]).T
        yhat = np.array([[0.2, 0.4, 0.2, 0.2], [0.5, 0.2, 0.2, 0.1],
                         [0.4, 0.1, 0.2, 0.9]]).T

        rel_error_val = cast(
            np.float64,
            rel_error(TestCrossEntropy.cross_entropy_object.get_loss(y, yhat),
                      np.float64(0.571599476)))

        self.assertLessEqual(float(rel_error_val), 1e-9)

    def test_cross_entropy_gradient1(self):
        y = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]).T
        yhat = np.array([[0.2, 0.4, 0.2, 0.2], [0.5, 0.2, 0.2, 0.1],
                         [0.4, 0.1, 0.2, 0.9]]).T
        jacobian_matrix = np.array([[0, -0.66666667, 0], [-0.83333333, 0, 0],
                                    [0, 0, 0], [0, 0, -0.37037037]])

        self.assertEqual(
            np.allclose(
                TestCrossEntropy.cross_entropy_object.get_gradient_pred(
                    y, yhat), jacobian_matrix), True)

    def test_forward2(self):
        y = np.array([0, 0, 1, 0]).T
        yhat = np.array([0.32574286, 0.081362, 0.0352241, 0.55767104]).T

        rel_error_val = cast(
            np.float64,
            rel_error(TestCrossEntropy.cross_entropy_object.get_loss(y, yhat),
                      np.float64(3.34602477)))
        self.assertTrue(rel_error_val <= 1e-9)

    def test_forward3(self):
        y = np.random.rand(35, 50)
        yhat = np.random.rand(35, 50)

        output = TestCrossEntropy.cross_entropy_object.get_loss(y, yhat)
        expected = np.mean(-np.sum(y * np.log(yhat), axis=0, keepdims=True))

        rel_error_val = cast(np.float64, rel_error(output, expected))
        self.assertTrue(rel_error_val <= 1e-9)

    def test_cross_entropy_gradient2(self):
        y = np.array([0, 0, 1, 0]).T
        yhat = np.array([0.32574286, 0.081362, 0.0352241, 0.55767104]).T
        jacobian_matrix = np.array([[0, 0, -28.38965367, 0]]).T
        gradient = TestCrossEntropy.cross_entropy_object.get_gradient_pred(
            y.reshape(-1, 1), yhat.reshape(-1, 1))

        rel_error_arr = rel_error(jacobian_matrix, gradient)
        self.assertTrue(np.all(rel_error_arr <= 1e-8))

    def test_backward1(self):
        # shape (C, m) where m is number of examples == 1 in this case
        # with 4 classes
        y = np.array([1500, 2500, 49000, 5012]).reshape(-1, 1)
        yhat = np.array([0.33, 0.33, 0.33, 0.01],
                        dtype=np.float64).reshape(-1, 1)
        rel_error_grad_array, rel_error_computed_vectors = (
            TestCrossEntropy.cross_entropy_object.gradient_checking(
                y, yhat, num_checks=2))

        self.assertTrue(np.all(rel_error_grad_array <= 8e-9))
        self.assertTrue(np.all(rel_error_computed_vectors <= 8e-9))

    def test_backward2(self):
        # shape (C, m) where m is number of examples == 15 in this case
        y = TestCrossEntropy.rs.rand(10, 15)
        yhat = TestCrossEntropy.rs.rand(10, 15)

        rel_error_grad_array, rel_error_computed_vectors = (
            TestCrossEntropy.cross_entropy_object.gradient_checking(
                y, yhat, num_checks=2))

        self.assertTrue(np.all(rel_error_grad_array <= 2e-6))
        self.assertTrue(np.all(rel_error_computed_vectors <= 2e-6))

    def test_backward3(self):
        y = TestCrossEntropy.rs.rand(25, 64)
        yhat = TestCrossEntropy.rs.rand(25, 64)

        rel_error_grad_array, rel_error_computed_vectors = (
            TestCrossEntropy.cross_entropy_object.gradient_checking(
                y, yhat, num_checks=2))

        rel_error_bool_grad = rel_error_grad_array <= 1e-6
        rel_error_vectors = rel_error_computed_vectors <= 1e-6
        dict_rel_error = get_unique_counts_array(rel_error_bool_grad)
        dict_rel_error_vectors = get_unique_counts_array(rel_error_vectors)
        # get percent rel_error less than 1e-6
        percent_error_lesseq_bound = get_percent(dict_rel_error[True],
                                                 dict_rel_error[False])

        percent_error_lesseq_bound_vectors = get_percent(
            dict_rel_error_vectors[True], dict_rel_error_vectors[False])

        self.assertTrue(percent_error_lesseq_bound >= 0.92)
        self.assertTrue(percent_error_lesseq_bound_vectors >= 0.93)

        self.assertTrue(np.all(rel_error_grad_array <= 1e-4))
        self.assertTrue(np.all(rel_error_computed_vectors <= 1e-4))

    def test_backward4(self):
        y = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0],
                      [0, 0, 1, 0]]).reshape(4, -1)
        yhat = np.array([[0.2, 0.4, 0.2, 0.2], [0.5, 0.2, 0.2, 0.1],
                         [0.4, 0.1, 0.2, 0.9], [0.33, 0.33, 0.1, 0.24],
                         [0.5, 0.2, 0.2, 0.1]]).reshape(4, -1)

        rel_error_grad_array, rel_error_computed_vectors = (
            TestCrossEntropy.cross_entropy_object.gradient_checking(
                y, yhat, num_checks=2))

        self.assertTrue(np.all(rel_error_grad_array <= 1e-8))
        self.assertTrue(np.all(rel_error_computed_vectors <= 1e-8))


class TestNegativeLogLoss(unittest.TestCase):
    """ This class contains unit tests for the negative log loss
    function """

    @classmethod
    def setUpClass(cls):
        cls.negative_log_loss = NegativeLogLoss()
        cls.rs = np.random.RandomState(32)

    def test_forward1(self):
        y = np.array([0, 1, 0, 1, 0, 0, 1]).reshape(1, -1)
        yhat = np.array([0.2, 0.8, 0.5, 0.3, 0.9, 0.9, 0.1]).reshape(1, -1)
        computed_loss = TestNegativeLogLoss.negative_log_loss.get_loss(y, yhat)
        expected_loss = np.mean(-(y * np.log(yhat) +
                                  (1 - y) * np.log(1 - yhat)))

        rel_error_val = rel_error(computed_loss, expected_loss)
        self.assertTrue(rel_error_val <= 1e-11)

    def test_forwad2(self):
        y = np.array([0, 1, 0]).reshape(1, -1)
        yhat = np.array([0.5, 0.2, 0.5]).reshape(1, -1)

        computed_loss = TestNegativeLogLoss.negative_log_loss.get_loss(y, yhat)

        rel_error_val = rel_error(computed_loss, np.float64(0.9985774245179969))

        self.assertTrue(rel_error_val <= 1e-11)

    def test_forward3(self):
        y = np.array([1, 1, 1, 0]).reshape(1, -1)
        yhat = np.array([0.33, 1, 1, 0]).reshape(1, -1)

        computed_loss = TestNegativeLogLoss.negative_log_loss.get_loss(y, yhat)

        rel_error_val = rel_error(computed_loss, np.float64(0.2771656562054028))

        self.assertTrue(rel_error_val <= 1e-11)


if __name__ == "__main__":
    unittest.main()
