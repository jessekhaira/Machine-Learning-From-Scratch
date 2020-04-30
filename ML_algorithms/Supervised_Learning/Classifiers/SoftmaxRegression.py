import numpy as np
from ML_algorithms.Neural_Net_Util.NeuralNetwork_Base import NeuralNetwork_Base
from ML_algorithms.Neural_Net_Util.LossFunctions import cross_entropy
from ML_algorithms.Neural_Net_Util.ActivationFunctions import Softmax
import copy 


class softmax_regression(NeuralNetwork_Base):
    """
    This class is a template you can use to create softmax regresssion objects. 
    The softmax regression object is used for multiclass classification tasks, where
    the output labels are one-hot encoded to be of shape (C,M). The object contains 
    one hidden layer with numClasses neurons using the softmax activation function.

    Parameters:
    -> inLayerNeuron (int): Integer representing how many features are at the input to the classifier
    -> regularization (str): Type of regularization to use. Either "L2" or "L1" is accepted.
    -> regParameter(int): Integer representing the strength of the regularization
    """
    def __init__(self, inLayerNeuron, numClasses, regularization = None, regParameter = None):
        costFunc = cross_entropy(regularization, regParameter)
        activFunc = Softmax()
        super(softmax_regression, self).__init__(costFunc, inLayerNeuron)
        self.add_layer(numClasses, activFunc, 1)
