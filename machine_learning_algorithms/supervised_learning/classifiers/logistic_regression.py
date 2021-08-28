""" This module contains code representing the logistic regression
supervised machine learning algorithm, used for the task of
binary classification.
"""
import numpy as np
from machine_learning_algorithms.neural_net_utility.neural_net_base import NeuralNetworkBase
from machine_learning_algorithms.neural_net_utility.loss_functions import NegativeLogLoss
from machine_learning_algorithms.neural_net_utility.activation_functions import Sigmoid
from typing import Literal, Union


class LogisticRegression(NeuralNetworkBase):
    """ This class represents the logistic regression supervised
    machine learning algorithm, used for the task of binary classification.

    Attributes:
        inLayerNeuron:
            Integer representing how many features are to be input to
            the classifier

        classificationThreshold:
            Floating point value to be applied to the predictions,
            to separate the positive class from the negative
            class when predicting, or None if not wanted

        regularization:
            String that should be either "L2" or "L1" indicating the
            type of regularization to use.

        regParameter:
            Floating point value representing the strength of the
            regularization
    """

    def __init__(self,
                 inLayerNeuron: int,
                 classificationThreshold: Union[None, float] = None,
                 regularization: Literal["L2", "L1"] = None,
                 regParameter: float = None):
        loss_obj = NegativeLogLoss(regularization, regParameter)
        activ = Sigmoid()
        super(LogisticRegression, self).__init__(input_features=inLayerNeuron,
                                                 lossFunction=loss_obj)
        # Logistic regression has one fully connected layer, with a
        # single neuron, with the sigmoid activation function
        self.add_layer(1, activ)
        self.classificationThreshold = classificationThreshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = self._forward_propagate(X)
        if self.classificationThreshold:
            return predictions >= self.classificationThreshold
        return predictions


class OneVsAllLogisticRegression(object):
    """ This class represents the one vs all logistic regresison algorithm.

    Attributes:
        num_classes:
            Integer representing the number of classes in your dataset

        num_in_neurons:
            Integer representing the number of features in your dataset

        num_epochs:
            Integer representing the number of epochs you would like to
            train your N objects for

        learn_rate:
            Floating point value representing the speed at which to update
            parameters during gradient descent
    """

    def __init__(self, num_classes: int, num_in_neurons: int, num_epochs: int,
                 learn_rate: float):
        self.model = []
        for _ in range(num_classes):
            self.model.append(LogisticRegression(num_in_neurons))
        self.datasets = []
        self.num_epochs = num_epochs
        self.learn_rate = learn_rate

    def fit(self, xtrain: np.ndarray, ytrain: np.ndarray) -> None:
        self._build_datasets(xtrain, ytrain)
        for i in range(len(self.model)):
            self.model[i].fit(self.datasets[i][0],
                              self.datasets[i][1],
                              num_epochs=self.num_epochs,
                              learn_rate=self.learn_rate)

    def _build_datasets(self, xtrain: np.ndarray, ytrain: np.ndarray) -> None:
        classes_data = np.unique(ytrain)
        for i in range(len(classes_data)):
            curr_class = classes_data[i]
            only_one_labelis1 = (ytrain == curr_class).astype(int)
            self.datasets.append((xtrain, only_one_labelis1))

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert x.shape[0] == self.model[0].num_input, (
            "Your new data has to have as many features as what you trained on")
        predictions = []
        for i in range(len(self.model)):
            predictions.append(self.model[i].predict(x))
        # Stack all predictions next to each other in a matrix so we
        # can easily get col vals for each row using np.argmax()
        # Predictions from each unit will be a (1,M) vector, so we
        # need to stack them up in rows and then get max row
        # val for each example (Class) by saying
        # np.argmax(axis=0)
        matrix_pred = np.row_stack((i for i in predictions))
        # matrix_pred should be of shape
        # (num_features in example, num examples * num predictors)
        assert matrix_pred.shape == (len(self.model), x.shape[1])
        final_output = np.argmax(matrix_pred, axis=0)
        assert final_output.shape == (x.shape[1],)
        return final_output
