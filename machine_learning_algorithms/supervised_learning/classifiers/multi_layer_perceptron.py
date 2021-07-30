""" This module contains code representing an implementation of the
multi layer perceptron algorithm for supervised learning """
from machine_learning_algorithms.neural_net_utility.neural_net_base import NeuralNetworkBase
from machine_learning_algorithms.neural_net_utility.loss_functions import negative_log_loss, cross_entropy, mean_squared_error


class MultiLayerPerceptron(NeuralNetworkBase):
    """ This class represents a multi-layer perceptron used for
    supervised learning. The user will have to add the number of
    layers desired to this layer accordingly.

    Attributes:
        typeSupervised:
            String that should either be "binary", "multiclass",
            or "regression", indicating the specific type of supervised
            learning task this algorithm is going to be used for

        numberInputFeatures:
            Integer representing the number of input features on the data
            the user will train the network on

        regularization:
            String that should either be "L1", "L2" or None indicating no 
            regularization should be used

        regParameter:
            Integer representing the strength of the regularization
    """

    def __init__(self,
                 typeSupervised,
                 numberInputFeatures,
                 regularization=None,
                 regParameter=None):
        if typeSupervised == "binary":
            loss_obj = negative_log_loss(regularization=regularization,
                                         regParameter=regParameter)
        elif typeSupervised == "multiclass":
            loss_obj = cross_entropy(regularization=regularization,
                                     regParameter=regParameter)
        else:
            loss_obj = mean_squared_error(regularization=regularization,
                                          regParameter=regParameter)
        super(MultiLayerPerceptron,
              self).__init__(loss_obj, input_features=numberInputFeatures)

    def predictMLP(self, X, classificationThreshold=None):
        # For binary classification, we need a classification threshold to seperate out the
        # pos class from the neg class
        if classificationThreshold:
            predictions = self._forward_propagate(X)
            return (predictions >= classificationThreshold).astype(int)
        else:
            return self.predict(X)
