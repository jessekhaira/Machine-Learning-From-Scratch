from machine_learning_algorithms.neural_net_utility.neural_net_base import NeuralNetworkBase
from machine_learning_algorithms.neural_net_utility.loss_functions import cross_entropy
from machine_learning_algorithms.neural_net_utility.activation_functions import Softmax


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

        regParameter:
            Floating point value representing the strength of the regularization
    """

    def __init__(self,
                 inLayerNeuron,
                 numClasses,
                 regularization=None,
                 regParameter=None):
        costFunc = cross_entropy(regularization, regParameter)
        activFunc = Softmax()
        super(SoftmaxRegression, self).__init__(costFunc, inLayerNeuron)
        self.add_layer(numClasses, activFunc, 1)
