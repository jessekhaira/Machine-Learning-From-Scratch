""" This module contains code representing a convolutional neural network """
from ML_algorithms.Neural_Net_Util.neural_net_base import NeuralNetworkBase
from ML_algorithms.Neural_Net_Util.LossFunctions import negative_log_loss, cross_entropy, mean_squared_error
from typing import Literal


class ConvolutionalNeuralNetwork(NeuralNetworkBase):
    """ This class represents a convolutional neural network used for supervised
    learning. The user will have to add the number of layers desired to this
    network.

    Attributes:
        typeSupervised:
            String indicating what kind of classification the network will
            carry out. Should be either "binary", "multiclass" or "regression".

        inputDepth:
            Integer representing the depth of the input volume

    """

    def __init__(self, typeSupervised: Literal["binary", "multiclass",
                                               "regression"], inputDepth: int):
        if typeSupervised == "binary":
            loss_obj = negative_log_loss()
        elif typeSupervised == "multiclass":
            loss_obj = cross_entropy()
        else:
            loss_obj = mean_squared_error()
        super(ConvolutionalNeuralNetwork, self).__init__(loss_obj,
                                                         input_features=None)
        self.inputDepth = inputDepth

    def addConvNetLayer(self, layer, **kwargs):
        """ This method adds a layer to a Convolutional Neural network object.

        Args:
            layer:
                A layer object that the user wants to add to the network
        """
        # We have no idea how many activated neurons we have in a conv layer
        # because that depends on what our input is, therefore we have to wait
        # to initialize the filters for every layer except the first layer
        inputDepth = self.inputDepth if not self.layers else None
        if "filterSize" in kwargs and "poolType" not in kwargs:
            self.layers.append(
                layer(filterSize=kwargs["filterSize"],
                      inputDepth=inputDepth,
                      numFilters=kwargs["numFilters"],
                      activationFunction=kwargs["activationFunction"],
                      padding=kwargs["padding"],
                      stride=kwargs["stride"],
                      finalConvLayer=kwargs["finalConvLayer"]))
        elif "filterSize" in kwargs and "poolType" in kwargs:
            self.layers.append(
                layer(filterSize=kwargs["filterSize"],
                      padding=kwargs["padding"],
                      stride=kwargs["stride"],
                      finalConvLayer=kwargs["finalConvLayer"],
                      poolType=kwargs["poolType"]))
        else:
            # adding densee layer - dealing with default arguments defined in the original add_layer method
            isSoftmax = 0 if "isSoftmax" not in kwargs else kwargs["isSoftmax"]
            keep_prob = 1 if "keep_prob" not in kwargs else kwargs["keep_prob"]

            self.add_layer(num_neurons=kwargs["num_neurons"],
                           activationFunction=kwargs["activationFunction"],
                           isSoftmax=isSoftmax,
                           layer=layer,
                           keep_prob=keep_prob)
