""" This module contains code representing a convolutional neural network """
from machine_learning_algorithms.neural_net_utility.neural_net_base import NeuralNetworkBase
from machine_learning_algorithms.neural_net_utility.loss_functions import NegativeLogLoss, CrossEntropy, MeanSquaredError
from typing import Literal


class ConvolutionalNeuralNetwork(NeuralNetworkBase):
    """ This class represents a convolutional neural network used for supervised
    learning. The user will have to add the number of layers desired to this
    network.

    Attributes:
        typeSupervised:
            String indicating what kind of classification the network will
            carry out. Should be either "binary", "multiclass" or "regression".

        input_depth:
            Integer representing the depth of the input volume

    """

    def __init__(self, typeSupervised: Literal["binary", "multiclass",
                                               "regression"], input_depth: int):
        if typeSupervised == "binary":
            loss_obj = NegativeLogLoss()
        elif typeSupervised == "multiclass":
            loss_obj = CrossEntropy()
        else:
            loss_obj = MeanSquaredError()
        super(ConvolutionalNeuralNetwork, self).__init__(loss_obj,
                                                         input_features=None)
        self.input_depth = input_depth

    def addConvNetLayer(self, layer, **kwargs):
        """ This method adds a layer to a Convolutional Neural network object.

        Args:
            layer:
                A layer object that the user wants to add to the network
        """
        # We have no idea how many activated neurons we have in a conv layer
        # because that depends on what our input is, therefore we have to wait
        # to initialize the filters for every layer except the first layer
        input_depth = self.input_depth if not self.layers else None
        if "filter_size" in kwargs and "poolType" not in kwargs:
            self.layers.append(
                layer(filter_size=kwargs["filter_size"],
                      input_depth=input_depth,
                      num_filters=kwargs["num_filters"],
                      activation_function=kwargs["activation_function"],
                      padding=kwargs["padding"],
                      stride=kwargs["stride"],
                      final_conv_layer=kwargs["final_conv_layer"]))
        elif "filter_size" in kwargs and "poolType" in kwargs:
            self.layers.append(
                layer(filter_size=kwargs["filter_size"],
                      padding=kwargs["padding"],
                      stride=kwargs["stride"],
                      final_conv_layer=kwargs["final_conv_layer"],
                      poolType=kwargs["poolType"]))
        else:
            # adding densee layer - dealing with default arguments defined in the original add_layer method
            isSoftmax = 0 if "isSoftmax" not in kwargs else kwargs["isSoftmax"]
            keep_prob = 1 if "keep_prob" not in kwargs else kwargs["keep_prob"]

            self.add_layer(num_neurons=kwargs["num_neurons"],
                           activation_function=kwargs["activation_function"],
                           isSoftmax=isSoftmax,
                           layer=layer,
                           keep_prob=keep_prob)
