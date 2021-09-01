""" This module contains code representing various different regression
algorithms, like lasso regression, ridge regression, and a base
class which they all inherit from """
import numpy as np
from machine_learning_algorithms.neural_net_utility.neural_net_base import NeuralNetworkBase
from machine_learning_algorithms.neural_net_utility.loss_functions import MeanSquaredError
from machine_learning_algorithms.neural_net_utility.activation_functions import IdentityActivation
from itertools import combinations_with_replacement
from machine_learning_algorithms.neural_net_utility.optimizer import GradientDescent, Optimizer
from sklearn import preprocessing
from typing import Union, Tuple


class BaseLinearRegression(NeuralNetworkBase):
    """ This class represents the base class for linear regression, which all
    linear regression classes will inherit.
    """

    def __init__(self, degree, regularization=None, reg_parameter=None):
        # Save the degree of the polynomial function that is desired to be fit.
        # Will be used later to transform the input features to the final
        # function features we will fit
        self.degree = degree
        # Loss function for regression tasks is RSS averaged over all
        # examples = MSE
        loss_function = MeanSquaredError(regularization, reg_parameter)
        super(BaseLinearRegression, self).__init__(lossFunction=loss_function,
                                                   input_features=None)

    def fit_iterative_optimizer(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        xvalid: Union[np.ndarray, None] = None,
        yvalid: Union[np.ndarray, None] = None,
        num_epochs: int = 10,
        batch_size: int = 32,
        ret_train_loss: bool = False,
        learn_rate: float = 0.01,
        optim: Optimizer = GradientDescent()
    ) -> Union[Tuple[int, int, int, int], Tuple[int, int], None]:
        """ This method learns the parameters of the algorithm by minimizing the
        residual sum of squares cost function using iterative Optimizers,
        which by default is set to gradient descent. Validation is done using
        xvalid and yvalid, if provided.

        Args:
            xtrain:
                Numpy array representing the feature vectors the algorithm will
                be training on

            ytrain:
                Numpy array representing the labels for xtrain

            xvalid:
                Numpy array representing the feature vectors the algorithm
                will be validated on, or None if not wanted

            yvalid:
                Numpy array representing the labels for xvalid, or None
                if not wanted. Should be None if xvalid is not specified.

            num_epochs:
                Integer representing the number of epochs to train the algorithm
                for

            batch_size:
                Integer representing the size of the minibatch to use when
                running the gradient descent optimization algorithm

            ret_train_loss:
                Boolean value representing whether or not to return the
                training loss

            learn_rate:
                Floating point value indicating the learning rate to use
                when running the gradient descent optimization algorithm

            optim:
                Object of type Optimizer, which is by default set to vanilla
                gradient descent.

        Returns:
            If ret_train_loss is set to True and xvalid is not None,
            4 integers will be returned in a tuple, indicating the training
            loss, validation loss, training accuracy, and validation accuracy
            respectively.

            If ret_train_loss is True and xvalid is None, then two integers
            will be returned in a tuple, indicating the training loss and
            training accuracy respectively.

            Otherwise, None will be returned.
        """
        # the fit method is basically the same as the neural net base, other
        # than the transformation of the features that needs to take place
        # before fitting
        xtrain = self._get_polynomial_features(xtrain)
        if xvalid is not None:
            xvalid = self._get_polynomial_features(xvalid)
        # Number of features is on the rows, so num_input == len(X_poly)
        self.num_input = len(xtrain)
        # Linear regression models have one layer with one neuron using an
        # identity activation function
        activ = IdentityActivation()
        self.add_layer(1, activ)
        # If ret_train_loss is true, we will return a list of the losses
        # averaged over each epoch for the training set and the
        # validation set

        return self.fit(xtrain,
                        ytrain,
                        xvalid=xvalid,
                        yvalid=yvalid,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        ret_train_loss=ret_train_loss,
                        learn_rate=learn_rate,
                        optim=optim)

    def predict_linear_regression(self, x: np.ndarray) -> float:
        # the predict method is basically the same as the neural net base,
        # other than the transformation of the features to the polynomial
        # features that needs to take place before fitting
        x_poly = self._get_polynomial_features(x)
        return self.predict(x_poly)

    def _get_polynomial_features(self, dataset: np.ndarray) -> np.ndarray:
        # Features on rows, examples on columns
        if self.degree == 1:
            return dataset
        original_num_features = len(dataset)
        # Get combinations of indices of features ex: (0,), (1,), (2,),
        # (0,0), (0,1), (0,2), (1,2), etc
        all_combos = self._get_combos(original_num_features, self.degree)
        num_polynomial_features = len(all_combos)
        # Make a empty new data matrix of the appropriate shape
        # We will fill in the rows of this matriix with the appropriate
        # feature values
        new_x = np.empty((num_polynomial_features, dataset.shape[1]))
        # Using the combo of the features which is a tuple like (0,0,0)
        # We can say np.prod(dataset[combo_features,:], axis=0) which will do
        # an element by element multiplication along the feature values in
        # each corresponding column
        for feature_idx, combo_features in enumerate(all_combos):
            new_x[feature_idx, :] = np.prod(dataset[combo_features, :], axis=0)
        # Preprocess your new dataset after you've made all the features
        # that you want to use
        new_x = preprocessing.scale(new_x.T)
        return new_x.T

    def _get_combos(self, num_features: int, degree: int):
        all_combos = [
            combinations_with_replacement(range(num_features), i)
            for i in range(1, degree + 1)
        ]
        unrolled = [item for sublist in all_combos for item in sublist]
        return unrolled


class LinearRegression(BaseLinearRegression):

    def __init__(self, degree):
        super(LinearRegression, self).__init__(degree=degree,
                                               regularization=None,
                                               reg_parameter=None)


class RidgeRegression(BaseLinearRegression):

    def __init__(self, degree, regParam=0.2):
        super(RidgeRegression, self).__init__(degree=degree,
                                              regularization="L2",
                                              reg_parameter=regParam)


class LassoRegression(BaseLinearRegression):

    def __init__(self, degree, regParam=0.2):
        super(LassoRegression, self).__init__(degree=degree,
                                              regularization="L1",
                                              reg_parameter=regParam)
