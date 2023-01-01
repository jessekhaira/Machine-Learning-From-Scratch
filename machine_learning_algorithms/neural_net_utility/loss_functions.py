""" This module contains a variety of objective functions """
import numpy as np
import random
from typing import Union, Literal, List, Any, Tuple, cast
from machine_learning_algorithms.neural_net_utility.neural_net_layers import BaseNeuralNetworkLayer
from machine_learning_algorithms.utility.misc import rel_error


class LossFunction:
    """ This is the base class which all loss functions will inherit from.

    Every loss function will have a method of get_loss,
    get_gradient_pred, and gradient_checking, therefore it made
    sense to make an abstract class from which all these related classes will
    inherit from.

    Attributes:
        regularization:
            String that is restricted to being either "L2" or "L1" indicating
            the type of regularization to be used, or None

        reg_parameter:
            Floating point value indicating the strength of the regularization
            if being used
    """

    def __init__(self,
                 regularization: Union[Literal["L1", "L2"], None] = None,
                 reg_parameter: Union[None, float] = None):
        self.regularization = regularization
        self.reg_parameter = reg_parameter

    def get_loss(self,
                 labels: np.ndarray,
                 predictions: np.ndarray,
                 layers_of_weights: Union[np.ndarray, None] = None):
        raise NotImplementedError

    def get_gradient_pred(self, labels: np.ndarray, predictions: np.ndarray):
        raise NotImplementedError

    def data_loss_with_regularization(
        self, data_loss: np.ndarray,
        layers_of_weights: Union[np.ndarray, List[BaseNeuralNetworkLayer]]
    ) -> np.float32:
        assert self.reg_parameter is not None, (
            "The regularization parameter must be specified" +
            "to get a regularized loss")
        reg_loss = 0
        if self.regularization == "L2":
            for i in range(len(layers_of_weights)):
                reg_loss += (np.linalg.norm(layers_of_weights[i].W, ord=2)**2)

        elif self.regularization == "L1":
            for i in range(len(layers_of_weights)):
                reg_loss += np.linalg.norm(layers_of_weights[i].W, ord=1)
        return np.mean(data_loss + self.reg_parameter * reg_loss)

    def gradient_checking(self,
                          labels: np.ndarray,
                          predictions: np.ndarray,
                          num_checks: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """ This method does a quick gradient check to ensure the
        dL/dA is indeed correct.

        Args:
            labels:
                Numpy array of shape (C, m), representing the labels
                for all input examples

            predictions:
                Numpy array of shape (C,m), representing the predictions
                for m examples

            num_checks:
                Integer representing the number of times to check the
                gradient implentation
        """
        eps = 1e-7
        output = np.zeros((num_checks, *predictions.shape))
        np.random.seed(32)
        # grad_analytic computes dJ/dpredictions for every single
        # prediction in vectorized fashion -- we are going to specifically
        # modify prediction @ every idx one at a time by upeps and downeps to
        # compute dJ/dPred_i and we can slice out of grad analytic
        # to compare them
        grad_analytic = self.get_gradient_pred(labels, predictions)
        grad_computed = np.zeros_like(grad_analytic)
        for l in range(num_checks):
            # multivariable scalar valued func -- hold all values constant
            # except one, which will be modified by a very small value eps,
            # in order to compute dJ/da_j, to get numerical gradient which
            # can be compared to corresponding analytic gradient computed above
            it = np.nditer(predictions, flags=["multi_index"])
            while not it.finished:
                i, j = it.multi_index
                it.iternext()
                p_upeps = np.copy(predictions)
                p_upeps[i, j] += eps
                p_downeps = np.copy(predictions)
                p_downeps[i, j] -= eps
                loss_upeps = self.get_loss(labels, p_upeps)
                loss_downeps = self.get_loss(labels, p_downeps)
                grad_numeric = (loss_upeps - loss_downeps) / (2 * eps)
                rel_error_computed = rel_error(grad_analytic[i, j],
                                               grad_numeric)
                output[l, i, j] = rel_error_computed

                # so we can compare vector to vector at the end
                if l == 0:
                    grad_computed[i, j] = grad_numeric

        # Shape (C, m)
        rel_error_analytic_numeric_vectors = cast(
            np.ndarray, rel_error(grad_analytic, grad_computed))
        return output, rel_error_analytic_numeric_vectors


class NegativeLogLoss(LossFunction):
    """ This class represents the negative log loss function,
    which is typically the cost function to be optimized
    in binary classification tasks.
    """

    def get_loss(
            self,
            labels: np.ndarray,
            predictions: np.ndarray,
            layers_of_weights: Union[np.ndarray, None] = None) -> np.float32:
        """ This method computes the loss for the predictions over the
        given labels, and adds a regularization loss as well if needed.

        Args:
            labels:
                Numpy array of shape (C,m), representing the labels for
                all of the inputs

            predictions:
                Numpy array of shape (C,m) representing predictions

        Returns:
            Floating point value representing the loss
        """
        assert labels.shape == predictions.shape, (
            "Somethings wrong, your labels have to be the same " +
            "shape as the predictions!")
        # Numerical stability issues -> we never want to take the log of 0
        # so we clip our predictions at a lowest val of 1e-10
        predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
        data_loss = -(labels * np.log(predictions) +
                      (1 - labels) * np.log(1 - predictions))
        if self.regularization and layers_of_weights:
            return self.data_loss_with_regularization(data_loss,
                                                      layers_of_weights)
        return np.mean(data_loss)

    def get_gradient_pred(self, labels: np.ndarray,
                          predictions: np.ndarray) -> np.ndarray:
        """ This method represents the derivative of the cost
        function with respect to the input value. This gradient
        is meant to be passed back in the circuit of the neural
        network, and if there is regularization, the regularization
        will be included when updating the weights of a certain layer.

        C - number of classes
        m - number of examples

        Args:
            labels:
                Numpy array of shape (C,m), representing the labels for
                all of the inputs

            predictions:
                Numpy array of shape (C,m) representing predictions

        Returns:
            Jacobian matrix of shape (C,m)
        """
        assert labels.shape == predictions.shape, (
            "Somethings wrong, your labels have to be the same shape " +
            "as the predictions!")
        predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
        # Include 1/batchsize term here for when we backprop
        dl_da = 1 / labels.shape[1] * ((predictions - labels) /
                                       (predictions * (1 - predictions)))
        return dl_da


class MeanSquaredError(LossFunction):
    """ This class represents the mean squared error loss,
    which is typically the cost function to be optimized
    in regression tasks.
    """

    def get_loss(
            self,
            labels: np.ndarray,
            predictions: np.ndarray,
            layers_of_weights: Union[np.ndarray, Any,
                                     None] = None) -> np.float32:
        """ This method computes the loss for the predictions over the
        given labels, and adds a regularization loss as well if needed.

        Args:
            labels:
                Numpy array of shape (C,m), representing the labels for
                all of the inputs

            predictions:
                Numpy array of shape (C,m) representing predictions

        Returns:
            Floating point value representing the loss
        """
        assert labels.shape == predictions.shape, (
            "Somethings wrong, your labels have to be the same shape " +
            "as the predictions!")
        data_loss = (1 / 2) * np.square(np.subtract(labels, predictions))
        if self.regularization and layers_of_weights:
            return self.data_loss_with_regularization(data_loss,
                                                      layers_of_weights)
        return np.mean(data_loss)

    def get_gradient_pred(self, labels: np.ndarray,
                          predictions: np.ndarray) -> np.ndarray:
        """ This method represents the derivative of the cost
        function with respect to the input value. This gradient
        is meant to be passed back in the circuit of the neural
        network, and if there is regularization, the regularization
        will be included when updating the weights of a certain layer.

        C - number of classes
        m - number of examples

        Args:
            labels:
                Numpy array of shape (C,m), representing the labels for
                all of the inputs

            predictions:
                Numpy array of shape (C,m) representing predictions

        Returns:
            Jacobian matrix of shape (C,m)
        """
        assert labels.shape == predictions.shape, (
            "Somethings wrong, your labels have to be the same shape " +
            "as the predictions!")
        dl_da = (1 / labels.shape[1]) * (predictions - labels)
        return dl_da


class CrossEntropy(LossFunction):
    """ This class represents the cross entropy loss, which is typically
    the cost function to be optimized in multiclass classification.

    This cost function relies on the input being a (C,m) probability
    distribution as the same shape as the labels, where C is the
    number of classes you have in your data and m is the number
    of examples.
    """

    def get_loss(
            self,
            labels: np.ndarray,
            predictions: np.ndarray,
            layers_of_weights: Union[np.ndarray, Any,
                                     None] = None) -> np.float32:
        """ This method computes the loss for the predictions over the
        given labels, and adds a regularization loss as well if needed.

        Args:
            labels:
                Numpy array of shape (C,m), representing the labels for
                all of the inputs

            predictions:
                Numpy array of shape (C,m) representing predictions

        Returns:
            Floating point value representing the loss
        """
        # Numerical stability issues -> we never want to take the log of 0
        # so we clip our predictions at a lowest val of 1e-10
        predictions = np.clip(predictions, 1e-10, None)
        data_loss = -(labels * np.log(predictions))
        if self.regularization and layers_of_weights:
            return self.data_loss_with_regularization(data_loss,
                                                      layers_of_weights)
        return np.mean(np.sum(data_loss, axis=0))

    def get_gradient_pred(self, labels: np.ndarray,
                          predictions: np.ndarray) -> np.ndarray:
        """ This method represents the derivative of the cost
        function with respect to the input value. This gradient
        is meant to be passed back in the circuit of the neural
        network, and if there is regularization, the regularization
        will be included when updating the weights of a certain layer.

        C - number of classes
        m - number of examples

        Args:
            labels:
                Numpy array of shape (C,m), representing the labels for
                all of the inputs

            predictions:
                Numpy array of shape (C,m) representing predictions

        Returns:
            Jacobian matrix of shape (C,m)
        """
        assert labels.shape == predictions.shape, (
            "Somethings wrong, your labels have to be the same shape as" +
            "the predictions!")
        # -1/m dont forget in gradient for minibatch gradient descent, averaging
        # gradients over m examples
        dl_da = -(1 / labels.shape[1]) * (labels / predictions)
        return dl_da


def construct_loss_object(type_supervised_learning: str, regularization,
                          reg_parameter) -> LossFunction:
    if type_supervised_learning == "binary":
        loss_obj = NegativeLogLoss(regularization=regularization,
                                   reg_parameter=reg_parameter)
    elif type_supervised_learning == "multiclass":
        loss_obj = CrossEntropy(regularization=regularization,
                                reg_parameter=reg_parameter)
    else:
        loss_obj = MeanSquaredError(regularization=regularization,
                                    reg_parameter=reg_parameter)
    return loss_obj
