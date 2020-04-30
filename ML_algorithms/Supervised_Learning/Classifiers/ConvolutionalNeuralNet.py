import sys
sys.path.append("/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Neural_Net_Util")
import numpy as np
from NeuralNetwork_Base import NeuralNetwork_Base
from LossFunctions import negative_log_loss
from ActivationFunctions import Sigmoid
from LossFunctions import negative_log_loss
from LossFunctions import cross_entropy
from LossFunctions import mean_squared_error
from ConvolutionalLayers import Conv2D
from ConvolutionalLayers import Pool
import copy 

class ConvolutionalNeuralNetwork(NeuralNetwork_Base):
    """
    This class represents a convolutional neural network used for supervised learning. The user will have
    to add the number of layers desired to this layer accordingly. 

    Parameters:
    -> typeSupervised(str): binary, multiclass, or regression
    -> inputDepth (int): Integer representing the depth of the input volume
    
    """
    def __init__(self, typeSupervised, inputDepth):
        if typeSupervised == "binary":
            loss_obj = negative_log_loss()
        elif typeSupervised == "multiclass":
            loss_obj = cross_entropy()
        else:
            loss_obj = mean_squared_error()
        super(ConvolutionalNeuralNetwork,self).__init__(loss_obj, input_features = None)
        self.inputDepth = inputDepth

    def addConvNetLayer(self, layer, **kwargs):
        """
        This method adds a layer to a Convolutional Neural network object. 

        Parameters:
        -> layer (obj): A layer object that the user wants to add to the network
        -> **kwargs: Key-worded arguments passing in values for all of the layers parameters
        """
        # We have no idea how many activated neurons we have in a conv layer because that depends on 
        # what our input is, therefore we have to wait to initialize the filters for every layer except
        # the first layer
        inputDepth = self.inputDepth if not self.layers else None
        if "filterSize" in kwargs and "poolType" not in kwargs:
            self.layers.append(layer(filterSize = kwargs["filterSize"], inputDepth = inputDepth, numFilters = kwargs["numFilters"], activationFunction = kwargs["activationFunction"], padding = kwargs["padding"], stride = kwargs["stride"], finalConvLayer=kwargs["finalConvLayer"]))
        elif "filterSize" in kwargs and "poolType" in kwargs:
            self.layers.append(layer(filterSize = kwargs["filterSize"], padding = kwargs["padding"], stride = kwargs["stride"], finalConvLayer=kwargs["finalConvLayer"], poolType = kwargs["poolType"]))
        else:
            # adding densee layer - dealing with default arguments defined in the original add_layer method 
            isSoftmax = 0 if "isSoftmax" not in kwargs else kwargs["isSoftmax"]
            keep_prob= 1 if "keep_prob" not in kwargs else kwargs["keep_prob"]

            self.add_layer(num_neurons = kwargs["num_neurons"], activationFunction = kwargs["activationFunction"], isSoftmax = isSoftmax, layer = layer, keep_prob= keep_prob)

