""" This module contains a variety of objective functions """
import numpy as np
import random
from typing import Union, Literal


def regularization_loss(layers_of_weights: np.ndarray, typeReg: str) -> float:
    reg_loss = 0
    if typeReg == "L2":
        for i in range(len(layers_of_weights)):
            reg_loss += (np.linalg.norm(layers_of_weights[i].W, ord=2)**2)

    elif typeReg == "L1":
        for i in range(len(layers_of_weights)):
            reg_loss += np.linalg.norm(layers_of_weights[i].W, ord=1)

    return reg_loss


class LossFunction(object):
    """ This is the base class which all loss functions will inherit from.

    Every loss function will have a method of get_loss,
    get_gradient_pred, and _gradient_checking, therefore it made
    sense to make an abstract class from which all these related classes will
    inherit from.
    """

    def get_loss(self, labels: np.ndarray, predictions: np.ndarray,
                 layers_of_weights: np.ndarray):
        raise NotImplementedError

    def get_gradient_pred(self, labels: np.ndarray, predictions: np.ndarray):
        raise NotImplementedError

    def _gradient_checking(self,
                           labels: np.ndarray,
                           predictions: np.ndarray,
                           num_checks: int = 10) -> None:
        """ This method does a quick gradient check to ensure the
        dL/dA is indeed correct.

        Theoretically, we should be able to compute the loss with
        respect to every single example at one time. It turns out
        that you lose precision when you do it that way, so you
        don't get appropriate results for when you compute the
        gradient.

        Specifically, np.log seems to not perform well in terms
        of precise accuracy when applied to an entire vector. Thus,
        we just compute, our loss with a single example at a time
        as this seems to preserve accuracy much better.

        Arguments:
            labels:
                Numpy array of shape (m,1), representing the labels
                for all input examples

            predictions:
                Numpy array of shape (m,1), representing the predictions
                (probability between 0 and 1) for m examples

            num_checks:
                Integer representing the number of times to check the
                gradient implentation
        """
        eps = 1e-7
        random.seed(561)
        for _ in range(num_checks):
            if labels.shape[0] > 1:
                # Reshape multiclass labels to be easier to work with
                labels = labels.reshape(1, -1)
                predictions = predictions.reshape(1, -1)
            change_idx = np.random.randint(0, labels.shape[1])
            y = labels[:, change_idx].reshape(1, 1)
            p = predictions[:, change_idx].reshape(1, 1)
            p_upeps = p + eps
            loss_higher = self.get_loss(y, p_upeps, None)
            p_downeps = p - eps
            loss_lower = self.get_loss(y, p_downeps, None)
            grad_analytic = self.get_gradient_pred(y, p)
            grad_numeric = (loss_higher - loss_lower) / (2 * eps)
            rel_error = abs(grad_analytic -
                            grad_numeric) / abs(grad_analytic + grad_numeric +
                                                eps)
            print('rel error is %s' % (rel_error))


class NegativeLogLoss(LossFunction):
    """ This class represents the negative log loss function,
    which is typically the cost function to be optimized
    in binary classification tasks.

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

    def get_loss(self, labels: np.ndarray, predictions: np.ndarray,
                 layers_of_weights: np.ndarray) -> float:
        """ This method computes the loss for the predictions over the
        given labels, and adds a regularization loss as well if needed.

        Arguments:
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
        # Cost is averaged overall all examples so we get
        # Tot_cost_batch = 1/m * (loss_examples_batch + reg_loss_batch)
        # Tot_cost_batch = (1/m) * loss_examples_batch + (1/m)*reg_loss_batch
        reg_loss = regularization_loss(layers_of_weights, self.regularization)
        if self.regularization == "L2":
            return np.mean(data_loss + (self.reg_parameter / 2) * reg_loss)

        # One examples loss, say zeroth, is:
        # -(y0*log(yhat0) + (1-y0)*log(1-yhat0) + lambda*(L1 norm or L2 norm))
        # The entire loss is this summed up over the entire vector
        # of predictions, and is done in a vectorized manner
        elif self.regularization == "L1":
            return np.mean(data_loss + self.reg_parameter * reg_loss)

        # no regularization, just return mean of data loss
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

        Arguments:
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

    def get_loss(self, labels: np.ndarray, predictions: np.ndarray,
                 layers_of_weights: np.ndarray) -> float:
        """ This method computes the loss for the predictions over the
        given labels, and adds a regularization loss as well if needed.

        Arguments:
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
        # Numerical stability issues -> we never want to take the log of 0 so we
        # clip our predictions at a lowest val of 1e-10
        data_loss = (1 / 2) * np.square(np.subtract(labels, predictions))
        # Cost is averaged overall all examples so we get
        # Tot_cost_batch = 1/m * (loss_examples_batch + reg_loss_batch)
        # Tot_cost_batch = (1/m) * loss_examples_batch + (1/m)*reg_loss_batch
        reg_loss = regularization_loss(layers_of_weights, self.regularization)
        if self.regularization == "L2":
            return np.mean(data_loss + (self.reg_parameter / 2) * reg_loss)

        # One examples loss, say zeroth, is:
        # -(y0*log(yhat0) + (1-y0)*log(1-yhat0) + lambda*(L1 norm or L2 norm))
        # The entire loss is this summed up over the entire vector of
        # predictions. This operations has beeen vectorized to allow
        # this to happen
        elif self.regularization == "L1":
            return np.mean(data_loss + self.reg_parameter * reg_loss)

        # no regularization, just return mean of data loss
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

        Arguments:
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
                 layers_of_weights: Union[None, np.ndarray] = None) -> float:
        """ This method computes the loss for the predictions over the
        given labels, and adds a regularization loss as well if needed.

        Arguments:
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
        predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
        data_loss = -(labels * np.log(predictions))
        # Cost is averaged overall all examples so we get
        # Tot_cost_batch = 1/m * (loss_examples_batch + reg_loss_batch)
        # Tot_cost_batch = (1/m) * loss_examples_batch + (1/m)*reg_loss_batch
        reg_loss = regularization_loss(layers_of_weights, self.regularization)
        if self.regularization == "L2":
            return np.mean(data_loss + (self.reg_parameter / 2) * reg_loss)

        # One examples loss, say zeroth, is:
        # -(y0*log(yhat0) + (1-y0)*log(1-yhat0) + lambda*(L1 norm or L2 norm))
        # The entire loss is this summed up over the entire vector of
        # predictions. This operations has beeen vectorized to allow
        # this to happen
        elif self.regularization == "L1":
            return np.mean(data_loss + self.reg_parameter * reg_loss)

        # sum up all the losses for every single example (column wise sum) and
        # then average them and return
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

        Arguments:
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
        # -1/m dont forget in gradient!
        dl_da = -(1 / labels.shape[1]) * (labels / predictions)
        return dl_da
