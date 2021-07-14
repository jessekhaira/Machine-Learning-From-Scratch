import numpy as np
import random
from machine_learning_algorithms.neural_net_utility.neural_net_layers import BaseNeuralNetworkLayer
import math
from machine_learning_algorithms.utility.misc import findRowColMaxElem
from typing import Literal, Union


class BaseConvolutionalLayer(BaseNeuralNetworkLayer):

    def _padInput(self, x, filterSize, padding):
        pad_h, pad_w = self._get_padding(filterSize, padding)
        # x has images first and channels last - dont pad those
        images_padded = np.pad(x, ((0, 0), (0, 0), pad_h, pad_w),
                               mode="constant",
                               constant_values=0)
        return images_padded

    def _get_padding(self, filterSize, padding):
        # valid means padding is zero
        if padding == "valid":
            return (0, 0), (0, 0)
        # otherwise assume same padding
        else:
            pad_h1 = int(math.floor((filterSize - 1) / 2))
            pad_h2 = int(math.ceil((filterSize - 1) / 2))
            pad_w1 = int(math.floor((filterSize - 1) / 2))
            pad_w2 = int(math.ceil((filterSize - 1) / 2))

            return (pad_h1, pad_h2), (pad_w1, pad_w2)


class Conv2D(BaseConvolutionalLayer):
    """ This class represents a 2D convolutional layer.

    Attributes:
        filterSize:
            Integer representing the size of the filters

        numFilters:
            Integer representing the number of filters in this
            layer (indicates how many feature detectors you have,
            and how many features you will be learning)

        padding:
            String that should be either "same" or "valid". "same"
            indicates no spatial downsampling will happen, "valid"
            indicates no padding at all

        stride:
            Integer representing how far the filter will slide each
            step

        finalConvLayer:
            Boolean value indicating whether this is the last convolutional
            layer before fully connected layers

        inputDepth:
            Integer representing the depth of the input to this layer, or
            None
    """

    def __init__(self,
                 filterSize: int,
                 numFilters: int,
                 activationFunction,
                 padding: Literal["same", "valid"] = "same",
                 stride: int = 1,
                 finalConvLayer: bool = False,
                 inputDepth: Union[None, int] = None):
        self.filterSize = filterSize
        self.numFilters = numFilters
        self.padding = padding
        self.stride = stride
        self.inputDepth = inputDepth
        self.filters, self.b = self._initializeWeights()
        self.activationFunction = activationFunction
        self.finalConvLayer = finalConvLayer
        self.optim = None

    def compute_forward(self, x, train=True):
        """ This method computes the forward propagation step through
        the convolutional layer. Expects x to be a (sizeBatch, depth,
        height, width) tensor. This step combines the convolution
        operation and the activation function into one.

        Args:
            x:
                Numpy array of shape (sizeBatch, depth, height, width)
                containing a batch of examples for the layer to process

            train:
                Boolean value indicating whether this layer is in training
                mode or validation mode

        Returns:
            A numpy array of shape (numExamples, numFilters, self.filterSize,
            self.filterSize, self.numFilters) representing the output
            from the layer
        """
        p = 0 if self.padding == "valid" else int((self.filterSize - 1) / 2)
        output_height = int((x.shape[2] - self.filterSize + 2 * p) /
                            (self.stride) + 1)
        outputWidth = int((x.shape[3] - self.filterSize + 2 * p) /
                          (self.stride) + 1)
        self.Z = np.zeros(
            (x.shape[0], self.numFilters, output_height, outputWidth))

        padded_input = self._padInput(x, self.filterSize, self.padding)
        self.Ain = padded_input
        for i in range(padded_input.shape[0]):
            image = padded_input[i, :, :, :]
            # have to pad image with zeros
            for filter in range(self.numFilters):
                filterWeights = self.filters[filter, :, :, :]
                bias = self.b[filter]
                curr_rowPic = 0
                curr_rowNeuron = -1
                # while you haven't seen all the height rows
                while curr_rowPic + self.filterSize <= image.shape[1]:
                    # while you haven't seen the full width of this row
                    curr_rowNeuron += 1
                    # reset the column to zero for every single row we are at
                    # for both the picture and neuron
                    curr_colPic = 0
                    curr_colNeuron = 0
                    while curr_colPic + self.filterSize <= image.shape[2]:
                        curr_imageSlice = image[:, curr_rowPic:curr_rowPic +
                                                self.filterSize,
                                                curr_colPic:curr_colPic +
                                                self.filterSize]
                        # for this image and this filter, we fill curr_rowNeuron and curr_colNeuron with the value shown
                        self.Z[i, filter, curr_rowNeuron,
                               curr_colNeuron] = np.sum(
                                   curr_imageSlice * filterWeights) + bias

                        # slide the filter horizontally with a step size equal to stride
                        curr_colPic += self.stride
                        # the next activated neuron in this row of the picture
                        curr_colNeuron += 1

                    # when your finished sliding horizontally, slide the filter down vertically with step size equal to stride
                    curr_rowPic += self.stride

        # apply activation function to the activation maps for every single image elementwise
        self.A = self.activationFunction.compute_output(self.Z)
        # if its the final conv layer, then return a flattened vector as output
        # otherwise, return it as is
        if not self.finalConvLayer:
            return self.A
        else:
            return self.A.reshape(-1, x.shape[0])

    def _initializeWeights(self):
        if self.inputDepth is None:
            return None, None
        # we are going to have F x F x D1 x K total filters in this layer
        filters = np.random.rand(self.numFilters, self.inputDepth,
                                 self.filterSize, self.filterSize) * 0.01
        # we have one bias term for each filter
        bias = np.zeros((self.numFilters, 1))
        return filters, bias

    def update_weights(self,
                       dLdA,
                       learn_rate,
                       epoch,
                       prediction_obj,
                       curr_x,
                       curr_y,
                       layer,
                       gradCheck=False):
        # if this is the last conv layer, then we reshape our output in the forward pass to be
        # a N features by M examples matrix

        # so when we're coming back, we have to reshape that to be (num ex, num dim, H, W) IE same shape as
        # the output of the layer. Basically reverse of the flattening to a single vector step :D
        if self.finalConvLayer:
            dLdA = dLdA.reshape(self.Z.shape[0], self.Z.shape[1],
                                self.Z.shape[2], self.Z.shape[3])

        # get dAdZ to get dLdZ
        dadz = self.activationFunction.getDerivative_wrtInput(self.Z)
        dLdZ = dLdA * dadz

        # going to fill in dLdW and dLdB and then update every weight in every filter with the optimizer
        dLdW = np.zeros_like(self.filters)
        dLdb = np.zeros((self.numFilters, 1, 1, 1))
        dLdA_prevLayer = np.zeros_like(self.Ain)

        for i in range(self.Ain.shape[0]):
            image = self.Ain[i, :, :, :]
            # get dLdW per each filter
            for filter in range(self.numFilters):
                curr_rowPic = 0
                curr_rowNeuron = -1
                while curr_rowPic + self.filterSize <= image.shape[1]:
                    curr_rowNeuron += 1
                    curr_colPic = 0
                    curr_colNeuron = 0
                    while curr_colPic + self.filterSize <= image.shape[2]:
                        # this image slice is responsible for creating a SINGLE neuron
                        # in other words, this is a multivariable scalar function as it takes multiple dimensions in
                        # and returns a single value

                        # we accumulate the gradients for the current filter over every part of the current image, AND over every single image that these
                        # filters see
                        curr_imageSlice = image[:, curr_rowPic:curr_rowPic +
                                                self.filterSize,
                                                curr_colPic:curr_colPic +
                                                self.filterSize]
                        neuron_gradientHere = dLdZ[i, filter, curr_rowNeuron,
                                                   curr_colNeuron]

                        dLdW[filter] += (curr_imageSlice * neuron_gradientHere)
                        dLdb[filter] += neuron_gradientHere

                        # for the ith picture, for every single dimension, for the current sliced out image
                        # we accumulate the gradients for the current input for every filter that sees the ith image
                        dLdA_prevLayer[i, :, curr_rowPic:curr_rowPic +
                                       self.filterSize,
                                       curr_colPic:curr_colPic +
                                       self.filterSize] += (
                                           self.filters[filter] *
                                           neuron_gradientHere)

                        curr_colPic += self.stride
                        curr_colNeuron += 1

                    curr_rowPic += self.stride

        # update filters and bias
        dLdb = np.sum(dLdb.reshape(-1, 1), axis=1, keepdims=True)
        self.filters, self.b = self.optim.updateParams([self.filters, self.b],
                                                       [dLdW, dLdb],
                                                       learn_rate,
                                                       epochNum=epoch + 1)

        # pass the gradient down the circuit
        return dLdA_prevLayer


class Pool(BaseConvolutionalLayer):
    """
    This class represents a pooling layer. The purpose of this class is to spatially downsample an input
    volume in terms of height and width while leaving the number of dimensions untouched. 

    Parameters:
    -> filterSize (int): The size of the filter you will be using to scan the input images 
    -> inputDepth (int): The depth of one slice in the input 
    -> padding (str): "same" or "valid". "same" indicates no spatial downsampling will happen,
    "valid" indicates no padding at all
    -> stride (int): Int representing how far the filter will slide each step
    -> poolType (str): The type of pooling that will be used at this layer. Should be max or avg. 
    """

    def __init__(self,
                 filterSize,
                 padding="valid",
                 stride=1,
                 poolType="max",
                 finalConvLayer=False):
        self.filterSize = filterSize
        self.typePool = poolType
        self.padding = padding
        self.stride = stride
        self.finalConvLayer = finalConvLayer
        self.optim = None

    def compute_forward(self, x, train=True):
        """
        This method computes the forward propagation step through the pool layer. 
        Expects x to be a (sizeBatch,depth, height, width) tensor. This step combines the convolution
        operation and the activation function into one.

        Parameters:
        -> x(sizeBatch, depth, height, width) NumPy tensor: Tensor that we are computing the forward pass through this layer
        on

        Returns: Output(numExamples, numFilters, self.filterSize, self.filterSize, self.numFilters) NumPy tensor.
        """
        p = 0 if self.padding is "valid" else (self.filterSize - 1) / 2
        output_height = int((x.shape[2] - self.filterSize + 2 * p) /
                            (self.stride) + 1)
        outputWidth = int((x.shape[3] - self.filterSize + 2 * p) /
                          (self.stride) + 1)
        self.Z = np.zeros((x.shape[0], x.shape[1], outputWidth, output_height))
        # technically it doesn't really make sense to pad in pool layers
        # but the option is still here
        padded_input = self._padInput(x, self.filterSize, self.padding)
        self.Ain = padded_input
        for i in range(padded_input.shape[0]):
            image = padded_input[i, :, :, :]
            # for every single image, you apply the spatial downsampling PER dimension
            # so we don't reduce the number of dimensions we have, we just reduce the height and width
            for dimension in range(image.shape[0]):
                curr_rowPic = 0
                curr_rowNeuron = -1
                # while you haven't seen all the height rows
                while curr_rowPic + self.filterSize <= image.shape[1]:
                    # while you haven't seen the full width of this row
                    curr_rowNeuron += 1
                    # reset the column to zero for every single row we are at
                    # for both the picture and neuron
                    curr_colPic = 0
                    curr_colNeuron = 0
                    while curr_colPic + self.filterSize <= image.shape[2]:
                        curr_imageSlice = image[dimension,
                                                curr_rowPic:curr_rowPic +
                                                self.filterSize,
                                                curr_colPic:curr_colPic +
                                                self.filterSize]
                        # max pool or average pool
                        if self.typePool == "max":
                            self.Z[i, dimension, curr_rowNeuron,
                                   curr_colNeuron] = np.max(curr_imageSlice)
                        else:
                            self.Z[i, dimension, curr_rowNeuron,
                                   curr_colNeuron] = np.average(curr_imageSlice)

                        # slide the filter horizontally with a step size equal to stride
                        curr_colPic += self.stride
                        # the next activated neuron in this row of the picture
                        curr_colNeuron += 1

                    # when your finished sliding horizontally, slide the filter down vertically with step size equal to stride
                    curr_rowPic += self.stride

        # return input that has been downsampled spatially
        # if its the last conv layer, then flatten to a vector
        # otherwise, return as is
        if not self.finalConvLayer:
            return self.Z
        else:
            return self.Z.reshape(-1, x.shape[0])

    def update_weights(self,
                       dLdA,
                       learn_rate,
                       epoch,
                       prediction_obj,
                       curr_x,
                       curr_y,
                       layer,
                       gradCheck=False):
        # Pooling layer has no weights to update but we still need to get
        # dL/d(A_layer-1) to pass back to the layer before

        # if this is the last conv layer, then we reshape our output in the forward pass to be
        # a N features by M examples matrix

        # so when we're coming back, we have to reshape that to be (num ex, num dim, H, W) IE same shape as
        # the output of the layer. Basically reverse of the flattening to a single vector step :D
        if self.finalConvLayer:
            dLdA = dLdA.reshape(self.Z.shape[0], self.Z.shape[1],
                                self.Z.shape[2], self.Z.shape[3])

        # get dAout_dAin and chain it together with dLdA_out to get dLdA_in to pass back
        dLdA_in = np.zeros((self.Ain.shape[0], self.Ain.shape[1],
                            self.Ain.shape[2], self.Ain.shape[3]))

        for i in range(self.Ain.shape[0]):
            image = self.Ain[i, :, :, :]
            for dimension in range(image.shape[0]):
                curr_rowPic = 0
                curr_rowNeuron = -1
                while curr_rowPic + self.filterSize <= image.shape[1]:
                    curr_rowNeuron += 1
                    curr_colPic = 0
                    curr_colNeuron = 0
                    while curr_colPic + self.filterSize <= image.shape[2]:
                        # this image slice is responsible for creating a SINGLE neuron
                        # in other words, this is a multivariable scalar function as it takes multiple dimensions in
                        # and returns a single value
                        curr_imageSlice = image[dimension,
                                                curr_rowPic:curr_rowPic +
                                                self.filterSize,
                                                curr_colPic:curr_colPic +
                                                self.filterSize]
                        # in max pool, dAout/dA_in is zero everywhere except the MAX idx
                        if self.typePool == "max":
                            maxIdx_row, maxIdx_col = findRowColMaxElem(
                                curr_imageSlice)
                            dLdA_in[i, dimension, maxIdx_row,
                                    maxIdx_col] += 1 * dLdA[i, dimension,
                                                            maxIdx_row,
                                                            maxIdx_col]
                        # in avg pool,  dAout/dA_in is 1/N everywhere, where N is the total number of neurons in the depth slice for the ith
                        # since all the neurons in the img slice just get averaged
                        else:
                            dLdA_in[i, dimension,
                                    curr_rowPic:curr_rowPic + self.filterSize,
                                    curr_colPic:curr_colPic +
                                    self.filterSize] += (
                                        1 / np.size(curr_imageSlice) *
                                        dLdA[i, dimension, curr_rowNeuron,
                                             curr_colNeuron])

                        curr_colPic += self.stride
                        curr_colNeuron += 1

                    curr_rowPic += self.stride

        return dLdA_in
