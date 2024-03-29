""" This module contains classes that represent optimization algorithms
used to optimize objective functions in machine learning
"""
import numpy as np
from typing import Union


class Optimizer(object):
    """ This class is an abstract class that contains methods
    that every optimization algorithm will implement """

    def update_params(self):
        raise NotImplementedError


class GradientDescent(Optimizer):
    """ This class represents the vanilla gradient descent
    optimization algorithm """

    def update_params(self,
                      params: np.ndarray,
                      dparams: np.ndarray,
                      learn_rate: float,
                      epoch_num: Union[None, int] = None):
        """ This method updates the numpy array passed in called params
        using the gradients contained in the numpy array dparams,
        utilizing the gradient descent formula:

        W = W - learn_rate * dparams

        Args:
            params:
                Argument of type np.ndarray representing the parameters inside
                of some machine learning algorithm used to compute some function

            dparams:
                Argument of type np.ndarray representing the gradient of some
                function with respect to the params argument

            learn_rate:
                Floating point value representing the learning rate used inside
                of gradient descent

            epoch_num:
                Argument of type integer or None representing the epoch number

        Returns:
            Numpy array containing the updated version of the params
        """
        for i in range(len(params)):
            params[i] = params[i] - learn_rate * dparams[i]
        return params


class GradientDescentMomentum(Optimizer):
    """ This class represents the gradient descent algorithm
    using momentum """

    def __init__(self, beta: float = 0.9):
        self.running_gradients = []
        self.beta = beta

    def update_params(self,
                      params: np.ndarray,
                      dparams: np.ndarray,
                      learn_rate: float,
                      epoch_num: Union[None, int] = None):
        # epoch zero, initialize running gradients for every single parameter
        # in this layer
        if not self.running_gradients:
            for i in range(len(params)):
                self.running_gradients.append(np.zeros_like(params[i]))

        for i in range(len(params)):
            self.running_gradients[i] = self.beta * self.running_gradients[
                i] + (1 - self.beta) * dparams[i]
            params[i] = params[i] - learn_rate * self.running_gradients[i]
        return params


class AdaGrad(Optimizer):
    """ This class represents the adagrad optimization algorithm """

    def __init__(self):
        self.running_gradients = []
        self.eps = 1e-8

    def update_params(self,
                      params: np.ndarray,
                      dparams: np.ndarray,
                      learn_rate: float,
                      epoch_num: Union[None, int] = None) -> np.ndarray:
        if not self.running_gradients:
            for i in range(len(params)):
                self.running_gradients.append(np.zeros_like(dparams[i]))

        for i in range(len(params)):
            # add square of dL/dparam to the running gradient for this param
            self.running_gradients[i] += np.power(dparams[i], 2)
            params[i] = params[i] - (learn_rate * dparams[i]) / (
                np.sqrt(self.running_gradients[i] + self.eps))
        return params


class RMSProp(Optimizer):
    """ This class represents the RMSProp optimization algorithm """

    def __init__(self, beta: float = 0.9, eps: float = 1e-8):
        self.running_gradients = []
        self.beta = beta
        self.eps = eps

    def update_params(self,
                      params: np.ndarray,
                      dparams: np.ndarray,
                      learn_rate: float,
                      epoch_num: Union[None, int] = None):
        # epoch zero, initialize running gradients for every single
        # parameter in this layer
        if not self.running_gradients:
            for i in range(len(params)):
                self.running_gradients.append(np.zeros_like(params[i]))

        for i in range(len(params)):
            self.running_gradients[i] = self.beta * self.running_gradients[
                i] + (1 - self.beta) * np.square(dparams[i])
            params[i] = params[i] - (learn_rate * dparams[i]) / (
                np.sqrt(self.running_gradients[i]) + self.eps)

        return params


class Adam(Optimizer):
    """ This class represents the Adam optimization algorithm """

    def __init__(self,
                 beta1: float = 0.9,
                 beta2: float = 0.9,
                 eps: float = 1e-8):
        self.running_gradients = []
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def update_params(self,
                      params: np.ndarray,
                      dparams: np.ndarray,
                      learn_rate: float,
                      epoch_num: Union[None, int] = None):
        # epoch zero, initialize running gradients for every single parameter
        # in this layer
        if not self.running_gradients:
            for i in range(len(params)):
                self.running_gradients.append(
                    [np.zeros_like(params[i]),
                     np.zeros_like(params[i])])

        for i in range(len(params)):
            momentum = self.running_gradients[i][0]
            adaptive_lr = self.running_gradients[i][1]
            dl_dparam = dparams[i]
            # update the running average vectors based on current dL/dparam
            momentum = self.beta1 * momentum + (1 - self.beta1) * dl_dparam
            adaptive_lr = self.beta2 * adaptive_lr + (
                1 - self.beta2) * np.power(dl_dparam, 2)
            # unbias estimates - important when epoch_num is low
            momentum = momentum / (1 - (self.beta1**epoch_num))
            adaptive_lr = adaptive_lr / (1 - (self.beta2**epoch_num))
            # finally, update params and save new adaptiveLearnRateVec
            # and momentumUpdateVec
            params[i] = params[i] - (learn_rate * momentum) / (
                np.sqrt(adaptive_lr) + self.eps)
            self.running_gradients[i][0] = momentum
            self.running_gradients[i][1] = adaptive_lr

        return params
