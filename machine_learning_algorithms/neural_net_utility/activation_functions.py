""" This module contains code for functionality related to activation
functions used inside of neural networks """
import numpy as np
import random
from typing import Union


class BaseActivationFunction(object):
    """ This is meant to be an abstract class that every single
    activation function will inherit from. Specifically, every
    activation function will be computing some output given some
    input x. Every function will have some derivative/partial
    derivative with respect to its input.

    Lastly, a quick gradCheck method will be implemented for
    every class to ensure the gradient calculation is correct.

    Thus, it made sense to make an abstract class which all the
    related classes will inherit from.
    """

    def compute_output(self, x: np.ndarray):
        raise NotImplementedError

    def get_derivative_wrt_input(self, x: np.ndarray):
        raise NotImplementedError

    def _gradient_checking(self, x: np.ndarray, num_checks: int = 10) -> None:
        """ This method does a quick gradient check to ensure the
        da/dz is indeed correct.

        Args:
            x:
                Numpy array of shape (m,1) representing the
                predictions (prob between 0 and 1) for m examples

            num_checks:
                Integer representing the number of times to check the
                gradient implentation
        """
        eps = 1e-5
        random.seed(21)
        for _ in range(num_checks):
            change_idx = random.randrange(0, len(x))
            x = x[change_idx, change_idx]
            x_upeps = x + eps
            activ_higher = self.compute_output(x_upeps)
            x_downeps = x - eps
            activ_lower = self.compute_output(x_downeps)
            grad_analytic = self.get_derivative_wrt_x(x)
            grad_numeric = (activ_higher - activ_lower) / (2 * eps)
            rel_error = abs(grad_analytic - grad_numeric) / abs(grad_analytic +
                                                                grad_numeric)
            print('rel error is %s' % (rel_error))


class Sigmoid(BaseActivationFunction):
    """ This is the sigmoid function. The input can be a scalar,
    vector, or matrix, this function will just apply the activation
    elementwise.
    """

    def compute_output(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def get_derivative_wrt_input(
            self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """ This method returns da/dz, which is the derivative of
        the sigmoid function with respect to the input value.

        Args:
            x:
                Scalar or matrix input to get da/dz for, where a is the
                activation produced by this function and z is the input
                value

        Returns:
            A scalar value or matrix representing da/dz, where a is the
            activation produced by this function and z is the input
            value
        """
        output = self.compute_output(x)
        return output * (1 - output)


class IdentityActivation(BaseActivationFunction):
    """ This class implements the identity function. The methods
    are self explanatory. This function is useful to use in the
    output layer of a neural network performing regression.

    Args:
        x:
            Floating point value or tensor. We just apply the identity
            function elementwise to the input
    """

    def compute_output(self, x: np.ndarray):
        return x

    def get_derivative_wrt_input(self, x: np.ndarray):
        return 1


class Softmax(BaseActivationFunction):
    """ This class implements the softmax function. This function is
    useful to use in the output layer of a neural network performing
    multiclass classification.

    Key thing to realize is: the input to the softmax is a R^n vector
    and the output is a R^n vector. Therefore we cannot just apply this
    function elementwise to the activations at the previous layer.
    We can take the exp() of each term elementwise, but we sum along each
    column vector to normalize the activations for that example.

    In addition, the gradient of the softmax cannot just be applied
    elementwise either, as the softmax is a vector valued function.
    That means its gradient is a matrix called a Jacobian matrix.
    """

    def compute_output(self, x: np.ndarray):
        # Numerically stable softmax - subtract max(x) input vector x
        # before computing softmax
        max_pred_per_ex = np.amax(x, axis=0)
        x -= max_pred_per_ex
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def get_derivative_wrt_input(self, a: np.ndarray):
        """ This function computes da/dZ, which will be chained
        together with dL/da to produce the gradient da/dZ.

        C - Number of neurons in the softmax layer

        Args:
            a:
                Vector of shape (C,1) representing one of the activations
                in the fully activated matrix

        Returns:
            A numpy array of shape (C,C) representing the Jacobian matrix
            for the function.
        """
        # Gradient softmax == Jacobian matrix (IE d(output_1)/d(input_1),
        # d(output_1)/d(input_2), d(output_1)/d(input_3)..)
        jacobian_mat = np.diagflat(a)
        # Jacobian matrix will have activations on diagonal where
        # i == j and zeros everywhere else.
        # The dot product will return a CxC matrix, which we can then use
        # to do an element by element subtraction
        return jacobian_mat - np.dot(a, a.T)

    def _gradient_checking(self, input, num_checks=10):
        raise NotImplementedError


class ReLU(BaseActivationFunction):
    """ This class represents the ReLU function. The input
    can be a scalar, vector, or matrix, as ReLU applies
    activations elementwise.
    """

    def compute_output(
            self, x: Union[int, float,
                           np.ndarray]) -> Union[int, float, np.ndarray]:
        """
        This function computes the ReLU function elementwise
        over the input x

        Args:
            x:
                Integer, floating point value or tensor

        Returns:
            Integer, floating point value or tensor. Just returns the same
            data type that was input
        """
        return np.maximum(0, x)

    def get_derivative_wrt_input(self, x: np.ndarray):
        """ This method returns da/dz, which is the gradient of the
        relu function with respect to the input value.

        Args:
            x:
                Integer, floating point value or tensor

        Returns:
            A value of the same data type that was input representing
            da/dz
        """
        return (x > 0) * 1


class TanH(BaseActivationFunction):
    """ This class represents the TanH function. The input can be a scalar,
    vector, or matrix, as TanH applies activations elementwise.
    """

    def compute_output(
            self, x: Union[int, float,
                           np.ndarray]) -> Union[int, float, np.ndarray]:
        """ This function computes the TanH function elementwise
        over the given input.

        Args:
            x:
                Integer, floating point value or tensor

        Returns:
            A value of the same data type that was input representing
            da/dz
        """
        return np.tanh(x)

    def get_derivative_wrt_input(self, x):
        """
        This method returns da/dz -> the gradient of the TanH function with respect
        to the input value. 

        Parameters:
        -> x (int, NumPy vector, or NumPy matrix) -> input to get da/dz for 

        Returns: da/dz (int, NumPy vector, or NumPy matrix) -> function will get da/dz elementwise and return
        the output accordingly
        """
        return 1 - np.square(self.compute_output(x))
