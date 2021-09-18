""" This module contains code for functionality related to convolutional
neural network layers """
import numpy as np
from machine_learning_algorithms.neural_net_utility.neural_net_layers import BaseNeuralNetworkLayer
import math
from machine_learning_algorithms.utility.misc import findRowColMaxElem
from typing import Literal, Union


class BaseConvolutionalLayer(BaseNeuralNetworkLayer):
    """ This class represents the base class from which convolutional neural
    network layers will inherit """

    def _pad_input(self, x: np.ndarray, filterSize: int, padding: int):
        pad_h, pad_w = self._get_padding(filterSize, padding)
        # x has images first and channels last - dont pad those
        images_padded = np.pad(x, ((0, 0), (0, 0), pad_h, pad_w),
                               mode="constant",
                               constant_values=0)
        return images_padded

    def _get_padding(self, filterSize: int, padding: int):
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
        self.filters, self.b = self._initialize_weights()
        self.activationFunction = activationFunction
        self.finalConvLayer = finalConvLayer
        self.optim = None

    def compute_forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
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
        output_width = int((x.shape[3] - self.filterSize + 2 * p) /
                           (self.stride) + 1)
        self.Z = np.zeros(
            (x.shape[0], self.numFilters, output_height, output_width))

        padded_input = self._pad_input(x, self.filterSize, self.padding)
        self.Ain = padded_input
        for i in range(padded_input.shape[0]):
            image = padded_input[i, :, :, :]
            # have to pad image with zeros
            for curr_filter in range(self.numFilters):
                filter_weights = self.filters[curr_filter, :, :, :]
                bias = self.b[curr_filter]
                curr_row_example = 0
                curr_row_neuron = -1
                # while you haven't seen all the height rows
                while curr_row_example + self.filterSize <= image.shape[1]:
                    # while you haven't seen the full width of this row
                    curr_row_neuron += 1
                    # reset the column to zero for every single row we are at
                    # for both the picture and neuron
                    curr_col_example = 0
                    curr_col_neuron = 0
                    while curr_col_example + self.filterSize <= image.shape[2]:
                        curr_image_slice = image[:, curr_row_example:
                                                 curr_row_example +
                                                 self.filterSize,
                                                 curr_col_example:
                                                 curr_col_example +
                                                 self.filterSize]
                        # for this image and this curr_filter, we fill
                        # curr_row_neuron and curr_col_neuron with the value
                        # shown
                        self.Z[i, curr_filter, curr_row_neuron,
                               curr_col_neuron] = np.sum(
                                   curr_image_slice * filter_weights) + bias

                        # slide the curr_filter horizontally with a step size
                        # equal to stride
                        curr_col_example += self.stride
                        # the next activated neuron in this row of the
                        # picture
                        curr_col_neuron += 1

                    # when your finished sliding horizontally, slide the
                    # curr_filter down vertically with step size equal
                    # to stride
                    curr_row_example += self.stride

        # apply activation function to the activation maps for every
        # single image elementwise
        self.A = self.activationFunction.compute_output(self.Z)
        # if its the final conv layer, then return a flattened vector
        # as output otherwise, return it as is
        if not self.finalConvLayer:
            return self.A
        else:
            return self.A.reshape(-1, x.shape[0])

    def _initialize_weights(self):
        if self.inputDepth is None:
            return None, None
        # we are going to have F x F x D1 x K total filters in this layer
        filters = np.random.rand(self.numFilters, self.inputDepth,
                                 self.filterSize, self.filterSize) * 0.01
        # we have one bias term for each filter
        bias = np.zeros((self.numFilters, 1))
        return filters, bias

    def update_weights(self,
                       dl_da: np.ndarray,
                       learn_rate: float,
                       epoch: int,
                       prediction_obj,
                       curr_x: np.ndarray,
                       curr_y: np.ndarray,
                       layer,
                       gradCheck: bool = False) -> np.ndarray:
        # if this is the last conv layer, then we reshape our output
        # in the forward pass to be a N features by M examples matrix

        # so when we're coming back, we have to reshape that to be
        # (num ex, num dim, H, W) IE same shape as the output of the layer.
        # Basically reverse of the flattening to a single vector step :D
        if self.finalConvLayer:
            dl_da = dl_da.reshape(self.Z.shape[0], self.Z.shape[1],
                                  self.Z.shape[2], self.Z.shape[3])

        # get dAdZ to get dLdZ
        da_dz = self.activationFunction.get_derivative_wrt_input(self.Z)
        dl_dz = dl_da * da_dz

        # going to fill in dLdW and dLdB and then update every weight in
        # every filter with the optimizer
        dl_dw = np.zeros_like(self.filters)
        dl_db = np.zeros((self.numFilters, 1, 1, 1))
        dl_da_prev = np.zeros_like(self.Ain)

        for i in range(self.Ain.shape[0]):
            image = self.Ain[i, :, :, :]
            # get dl_dw per each filter
            for curr_filter in range(self.numFilters):
                curr_row_pic = 0
                curr_row_neuron = -1
                while curr_row_pic + self.filterSize <= image.shape[1]:
                    curr_row_neuron += 1
                    curr_col_pic = 0
                    curr_col_neuron = 0
                    while curr_col_pic + self.filterSize <= image.shape[2]:
                        # this image slice is responsible for creating a SINGLE
                        # neuron. In other words, this is a multivariable scalar
                        # function as it takes multiple dimensions in and
                        # returns a single value

                        # we accumulate the gradients for the current
                        # curr_filter over every part of the current image,
                        # AND over every single image that these filters see
                        curr_img_slice = image[:, curr_row_pic:curr_row_pic +
                                               self.filterSize,
                                               curr_col_pic:curr_col_pic +
                                               self.filterSize]
                        neuron_gradient_here = dl_dz[i, curr_filter,
                                                     curr_row_neuron,
                                                     curr_col_neuron]

                        dl_dw[curr_filter] += (curr_img_slice *
                                               neuron_gradient_here)
                        dl_db[curr_filter] += neuron_gradient_here

                        # for the ith picture, for every single dimension, for
                        # the current sliced out image we accumulate the
                        # gradients for the current input for every curr_filter
                        # that sees the ith image
                        dl_da_prev[i, :,
                                   curr_row_pic:curr_row_pic + self.filterSize,
                                   curr_col_pic:curr_col_pic +
                                   self.filterSize] += (
                                       self.filters[curr_filter] *
                                       neuron_gradient_here)

                        curr_col_pic += self.stride
                        curr_col_neuron += 1

                    curr_row_pic += self.stride

        # update filters and bias
        dl_db = np.sum(dl_db.reshape(-1, 1), axis=1, keepdims=True)
        self.filters, self.b = self.optim.update_params([self.filters, self.b],
                                                        [dl_dw, dl_db],
                                                        learn_rate,
                                                        epoch_num=epoch + 1)

        # pass the gradient down the circuit
        return dl_da_prev


class Pool(BaseConvolutionalLayer):
    """ This class represents a pooling layer. The purpose of this class
    is to spatially downsample an input volume in terms of height and
    width while leaving the number of dimensions untouched.

    Attributes:
        filterSize:
            Integer representing the size of the filter you will be
            using to scan the input images

        inputDepth:
            Integer representing the depth of one slice in the input

        padding:
            String that is either "same" or "valid". "same" indicates
            no spatial downsampling will happen, "valid" indicates no
            padding at all

        stride:
            Integer representing how far the filter will slide each step

        poolType:
            String that is either "max" or "avg", representing the type of
            pooling that will be used at this layer
    """

    def __init__(self,
                 filterSize: int,
                 padding: Literal["valid", "same"] = "valid",
                 stride: int = 1,
                 poolType: Literal["max", "avg"] = "max",
                 finalConvLayer: bool = False):
        self.filterSize = filterSize
        self.typePool = poolType
        self.padding = padding
        self.stride = stride
        self.finalConvLayer = finalConvLayer
        self.optim = None

    def compute_forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        """ This method computes the forward propagation step
        through the pool layer. Expects x to be a (sizeBatch, depth,
        height, width) tensor. This step combines the convolution
        operation and the activation function into one.

        Args:
            x:
                Numpy array of shape (sizeBatch, depth, height, width) that
                we are computing the forward pass through this layer on

        Returns:
            Numpy array of shape (numExamples, numFilters, self.filterSize,
            self.filterSize, self.numFilters) representing the spatially
            downsampled input
        """
        p = 0 if self.padding == "valid" else (self.filterSize - 1) / 2
        output_height = int((x.shape[2] - self.filterSize + 2 * p) /
                            (self.stride) + 1)
        output_width = int((x.shape[3] - self.filterSize + 2 * p) /
                           (self.stride) + 1)
        self.Z = np.zeros((x.shape[0], x.shape[1], output_width, output_height))
        # technically it doesn't really make sense to pad in pool layers
        # but the option is still here
        padded_input = self._pad_input(x, self.filterSize, self.padding)
        self.Ain = padded_input
        for i in range(padded_input.shape[0]):
            image = padded_input[i, :, :, :]
            # for every single image, you apply the spatial downsampling
            # PER dimension so we don't reduce the number of dimensions
            # we have, we just reduce the height and width
            for dimension in range(image.shape[0]):
                curr_row_pic = 0
                curr_row_neuron = -1
                # while you haven't seen all the height rows
                while curr_row_pic + self.filterSize <= image.shape[1]:
                    # while you haven't seen the full width of this row
                    curr_row_neuron += 1
                    # reset the column to zero for every single row we are at
                    # for both the picture and neuron
                    curr_col_pic = 0
                    curr_col_neuron = 0
                    while curr_col_pic + self.filterSize <= image.shape[2]:
                        curr_img_slice = image[dimension,
                                               curr_row_pic:curr_row_pic +
                                               self.filterSize,
                                               curr_col_pic:curr_col_pic +
                                               self.filterSize]
                        # max pool or average pool
                        if self.typePool == "max":
                            self.Z[i, dimension, curr_row_neuron,
                                   curr_col_neuron] = np.max(curr_img_slice)
                        else:
                            self.Z[i, dimension, curr_row_neuron,
                                   curr_col_neuron] = np.average(curr_img_slice)

                        # slide the filter horizontally with a step size equal
                        # to stride
                        curr_col_pic += self.stride
                        # the next activated neuron in this row of the
                        # picture
                        curr_col_neuron += 1

                    # when your finished sliding horizontally, slide the
                    # filter down vertically with step size equal to
                    # stride
                    curr_row_pic += self.stride

        # return input that has been downsampled spatially
        # if its the last conv layer, then flatten to a vector
        # otherwise, return as is
        if not self.finalConvLayer:
            return self.Z
        else:
            return self.Z.reshape(-1, x.shape[0])

    def update_weights(self,
                       dLdA: np.ndarray,
                       learn_rate: float,
                       epoch: int,
                       prediction_obj,
                       curr_x: np.ndarray,
                       curr_y: np.ndarray,
                       layer,
                       gradCheck: bool = False):
        # Pooling layer has no weights to update but we still
        # need to get dL/d(A_layer-1) to pass back to the layer
        # before

        # if this is the last conv layer, then we reshape our output in
        # the forward pass to be a N features by M examples matrix

        # so when we're coming back, we have to reshape that to be
        # (num ex, num dim, H, W) IE same shape as the output of the
        # layer. Basically reverse of the flattening to a single vector
        # step :D
        if self.finalConvLayer:
            dLdA = dLdA.reshape(self.Z.shape[0], self.Z.shape[1],
                                self.Z.shape[2], self.Z.shape[3])

        # get dAout_dAin and chain it together with dLdA_out to get
        # dl_da_in to pass back
        dl_da_in = np.zeros((self.Ain.shape[0], self.Ain.shape[1],
                             self.Ain.shape[2], self.Ain.shape[3]))

        for i in range(self.Ain.shape[0]):
            image = self.Ain[i, :, :, :]
            for dimension in range(image.shape[0]):
                curr_row_pic = 0
                curr_row_neuron = -1
                while curr_row_pic + self.filterSize <= image.shape[1]:
                    curr_row_neuron += 1
                    curr_col_pic = 0
                    curr_col_neuron = 0
                    while curr_col_pic + self.filterSize <= image.shape[2]:
                        # this image slice is responsible for creating a
                        # SINGLE neuron. In other words, this is a multivariable
                        # scalar function as it takes multiple dimensions in
                        # and returns a single value
                        curr_img_slice = image[dimension,
                                               curr_row_pic:curr_row_pic +
                                               self.filterSize,
                                               curr_col_pic:curr_col_pic +
                                               self.filterSize]
                        # in max pool, dAout/dA_in is zero everywhere except the
                        # MAX idx
                        if self.typePool == "max":
                            max_idx_row, max_idx_col = findRowColMaxElem(
                                curr_img_slice)
                            dl_da_in[i, dimension, max_idx_row,
                                     max_idx_col] += 1 * dLdA[i, dimension,
                                                              max_idx_row,
                                                              max_idx_col]
                        # in avg pool,  dAout/dA_in is 1/N everywhere, where N
                        # is the total number of neurons in the depth slice for
                        # the ith since all the neurons in the img slice just
                        # get averaged
                        else:
                            dl_da_in[i, dimension, curr_row_pic:curr_row_pic +
                                     self.filterSize,
                                     curr_col_pic:curr_col_pic +
                                     self.filterSize] += (
                                         1 / np.size(curr_img_slice) *
                                         dLdA[i, dimension, curr_row_neuron,
                                              curr_col_neuron])

                        curr_col_pic += self.stride
                        curr_col_neuron += 1

                    curr_row_pic += self.stride

        return dl_da_in
