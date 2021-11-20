""" This module contains tests for the softmax regression algorithm """
import unittest
import numpy as np
from machine_learning_algorithms.supervised_learning.classifiers.softmax_regression import SoftmaxRegression
from machine_learning_algorithms.neural_net_utility.optimizer import Adam, RMSProp
from machine_learning_algorithms.utility.score_functions import accuracy
from machine_learning_algorithms.utility.misc import one_hot_encode
from machine_learning_algorithms.utility.k_fold_cross_validation import k_fold_CV
from sklearn.datasets import load_iris
from sklearn import preprocessing


class SoftmaxTests(unittest.TestCase):
    """ This class contains unit tests for the softmax regression
    algorithm.

    A few notes:
        With L1 regularization and L2 regularization, the classifier performs
        as expected -> performance is very sensitive to reg_parameter. If the
        regularization parameter is even slightly high (>0.3), the performance
        for the l1 regularized and l2 regularized softmax regression models
        falter heavily.
    """

    def setUp(self):
        self.x, self.y = load_iris(return_X_y=True)
        self.x = preprocessing.scale(self.x).T
        self.y_encoded = one_hot_encode(self.y)
        self.softmax_model_no_regularization = SoftmaxRegression(
            self.x.shape[0], len(self.y_encoded))

        self.softmax_model_l1_regularization = SoftmaxRegression(
            self.x.shape[0],
            len(self.y_encoded),
            regularization="L1",
            reg_parameter=0.01)

        self.softmax_model_l2_regularization = SoftmaxRegression(
            self.x.shape[0],
            len(self.y_encoded),
            regularization="L2",
            reg_parameter=0.01)

        self.k_fold_obj = k_fold_CV()

    def test_softmax_no_reg(self):

        # Strength of RMSProp shown - get a 6% increase in accuracy w/ it. 99.3%
        # RMSprop and 93.7% normal gradient descent
        cross_val_score_gradient_descent = self.k_fold_obj.getKScore(
            self.x,
            self.y_encoded,
            accuracy,
            self.softmax_model_no_regularization,
            numEpochs=100,
            learn_rate=0.2,
            k=8)

        cross_val_score_rms_prop = self.k_fold_obj.getKScore(
            self.x,
            self.y_encoded,
            accuracy,
            self.softmax_model_no_regularization,
            numEpochs=100,
            learn_rate=0.2,
            k=8,
            optim=RMSProp())

        # Adam is the most sensitive out of the three tested and requires the
        # most hyperparameter tuning
        _, train_acc = self.softmax_model_no_regularization.fit(
            self.x,
            self.y_encoded,
            num_epochs=1000,
            learn_rate=0.01,
            optim=Adam(),
            ret_train_loss=True)

        cross_val_score_adam = self.k_fold_obj.getKScore(
            self.x,
            self.y_encoded,
            accuracy,
            self.softmax_model_no_regularization,
            numEpochs=1000,
            learn_rate=0.01,
            k=8,
            optim=Adam())

        self.assertGreaterEqual(np.average(train_acc), 0.90)
        self.assertGreaterEqual(cross_val_score_gradient_descent, 0.90)
        self.assertGreaterEqual(cross_val_score_rms_prop, 0.96)
        self.assertGreaterEqual(cross_val_score_adam, 0.85)

    def test_softmax_reg(self):

        cross_val_score_l1_reg = self.k_fold_obj.getKScore(
            self.x,
            self.y_encoded,
            accuracy,
            self.softmax_model_l1_regularization,
            numEpochs=150,
            learn_rate=0.01,
            k=8)

        cross_val_score_l2_reg = self.k_fold_obj.getKScore(
            self.x,
            self.y_encoded,
            accuracy,
            self.softmax_model_l2_regularization,
            numEpochs=150,
            learn_rate=0.01,
            k=8)

        self.assertGreaterEqual(cross_val_score_l1_reg, 0.80)
        self.assertGreaterEqual(cross_val_score_l2_reg, 0.80)


if __name__ == "__main__":
    unittest.main()
