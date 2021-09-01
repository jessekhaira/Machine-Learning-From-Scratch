""" This module contains code representing the softmax regression
multi class classification algorithm """
from machine_learning_algorithms.neural_net_utility.neural_net_base import NeuralNetworkBase
from machine_learning_algorithms.neural_net_utility.loss_functions import CrossEntropy
from machine_learning_algorithms.neural_net_utility.activation_functions import Softmax
from typing import Union, Literal


class SoftmaxRegression(NeuralNetworkBase):
    """ This class represents the softmax regression
    algorithm used for the task of multi class
    classification, where the output labels are one-hot
    encoded to be of shape (C,M). The object contains
    one hidden layer with numClasses neurons using the
    softmax activation function.

    Attributes:
        inLayerNeuron:
            Integer representing how many features are at the
            input to the classifier

        regularization:
            Value that is either a string type that is either "L2" or "L1"
            or a value of None type, representing the type of regularization
            to use

        reg_parameter:
            Floating point value representing the strength of the
            regularization, or None if regularization is not used
    """

    def __init__(self,
                 inLayerNeuron: int,
                 numClasses: int,
                 regularization: Union[Literal["L2", "L1"], None] = None,
                 reg_parameter: Union[float, None] = None):
        cost_func = CrossEntropy(regularization, reg_parameter)
        activ_func = Softmax()
        super(SoftmaxRegression, self).__init__(cost_func, inLayerNeuron)
        self.add_layer(numClasses, activ_func, 1)
