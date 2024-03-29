""" This module contains code for functionality related to convolutional
neural network layers """
import numpy as np
from machine_learning_algorithms.neural_net_utility.neural_net_layers import BaseNeuralNetworkLayer
import math
from machine_learning_algorithms.utility.misc import find_row_column_max_element
from typing import Literal, Union


class BaseConvolutionalLayer(BaseNeuralNetworkLayer):
    """ This class represents the base class from which convolutional neural
    network layers will inherit """

    def _pad_input(self, x: np.ndarray, filter_size: int, padding: int):
        pad_h, pad_w = self._get_padding(filter_size, padding)
        # x has images first and channels last - dont pad those
        images_padded = np.pad(x, ((0, 0), (0, 0), pad_h, pad_w),
                               mode="constant",
                               constant_values=0)
        return images_padded

    def _get_padding(self, filter_size: int, padding: int):
        # valid means padding is zero
        if padding == "valid":
            return (0, 0), (0, 0)
        # otherwise assume same padding
        else:
            pad_h1 = int(math.floor((filter_size - 1) / 2))
            pad_h2 = int(math.ceil((filter_size - 1) / 2))
            pad_w1 = int(math.floor((filter_size - 1) / 2))
            pad_w2 = int(math.ceil((filter_size - 1) / 2))

            return (pad_h1, pad_h2), (pad_w1, pad_w2)


class Conv2D(BaseConvolutionalLayer):
    """ This class represents a 2D convolutional layer.

    Attributes:
        filter_size:
            Integer representing the size of the filters

        num_filters:
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

        final_conv_layer:
            Boolean value indicating whether this is the last convolutional
            layer before fully connected layers

        input_depth:
            Integer representing the depth of the input to this layer, or
            None
    """

    def __init__(self,
                 filter_size: int,
                 num_filters: int,
                 activation_function,
                 padding: Literal["same", "valid"] = "same",
                 stride: int = 1,
                 final_conv_layer: bool = False,
                 input_depth: Union[None, int] = None):
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.padding = padding
        self.stride = stride
        self.input_depth = input_depth
        self.filters, self.b = self._initialize_weights()
        self.activation_function = activation_function
        self.final_conv_layer = final_conv_layer
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
            A numpy array of shape (numExamples, num_filters, self.filter_size,
            self.filter_size, self.num_filters) representing the output
            from the layer
        """
        p = 0 if self.padding == "valid" else int((self.filter_size - 1) / 2)
        output_height = int((x.shape[2] - self.filter_size + 2 * p) /
                            (self.stride) + 1)
        output_width = int((x.shape[3] - self.filter_size + 2 * p) /
                           (self.stride) + 1)
        self.z = np.zeros(
            (x.shape[0], self.num_filters, output_height, output_width))

        padded_input = self._pad_input(x, self.filter_size, self.padding)
        self.ain = padded_input
        for i in range(padded_input.shape[0]):
            image = padded_input[i, :, :, :]
            # have to pad image with zeros
            for curr_filter in range(self.num_filters):
                filter_weights = self.filters[curr_filter, :, :, :]
                bias = self.b[curr_filter]
                curr_row_example = 0
                curr_row_neuron = -1
                # while you haven't seen all the height rows
                while curr_row_example + self.filter_size <= image.shape[1]:
                    # while you haven't seen the full width of this row
                    curr_row_neuron += 1
                    # reset the column to zero for every single row we are at
                    # for both the picture and neuron
                    curr_col_example = 0
                    curr_col_neuron = 0
                    while curr_col_example + self.filter_size <= image.shape[2]:
                        curr_image_slice = image[:, curr_row_example:
                                                 curr_row_example +
                                                 self.filter_size,
                                                 curr_col_example:
                                                 curr_col_example +
                                                 self.filter_size]
                        # for this image and this curr_filter, we fill
                        # curr_row_neuron and curr_col_neuron with the value
                        # shown
                        self.z[i, curr_filter, curr_row_neuron,
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
        self.aout = self.activation_function.compute_output(self.z)
        # if its the final conv layer, then return a flattened vector
        # as output otherwise, return it as is
        if not self.final_conv_layer:
            return self.aout
        else:
            return self.aout.reshape(-1, x.shape[0])

    def _initialize_weights(self):
        if self.input_depth is None:
            return None, None
        # we are going to have F x F x D1 x K total filters in this layer
        filters = np.random.rand(self.num_filters, self.input_depth,
                                 self.filter_size, self.filter_size) * 0.01
        # we have one bias term for each filter
        bias = np.zeros((self.num_filters, 1))
        return filters, bias

    def update_weights(self,
                       dl_da: np.ndarray,
                       learn_rate: float,
                       epoch: int,
                       prediction_obj,
                       curr_x: np.ndarray,
                       curr_y: np.ndarray,
                       layer,
                       grad_check: bool = False) -> np.ndarray:
        # if this is the last conv layer, then we reshape our output
        # in the forward pass to be a N features by M examples matrix

        # so when we're coming back, we have to reshape that to be
        # (num ex, num dim, H, W) IE same shape as the output of the layer.
        # Basically reverse of the flattening to a single vector step :D
        if self.final_conv_layer:
            dl_da = dl_da.reshape(self.z.shape[0], self.z.shape[1],
                                  self.z.shape[2], self.z.shape[3])

        # get dAdZ to get dLdZ
        da_dz = self.activation_function.get_derivative_wrt_input(self.z)
        dl_dz = dl_da * da_dz

        # going to fill in dLdW and dLdB and then update every weight in
        # every filter with the optimizer
        dl_dw = np.zeros_like(self.filters)
        dl_db = np.zeros((self.num_filters, 1, 1, 1))
        dl_da_prev = np.zeros_like(self.ain)

        for i in range(self.ain.shape[0]):
            image = self.ain[i, :, :, :]
            # get dl_dw per each filter
            for curr_filter in range(self.num_filters):
                curr_row_pic = 0
                curr_row_neuron = -1
                while curr_row_pic + self.filter_size <= image.shape[1]:
                    curr_row_neuron += 1
                    curr_col_pic = 0
                    curr_col_neuron = 0
                    while curr_col_pic + self.filter_size <= image.shape[2]:
                        # this image slice is responsible for creating a SINGLE
                        # neuron. In other words, this is a multivariable scalar
                        # function as it takes multiple dimensions in and
                        # returns a single value

                        # we accumulate the gradients for the current
                        # curr_filter over every part of the current image,
                        # AND over every single image that these filters see
                        curr_img_slice = image[:, curr_row_pic:curr_row_pic +
                                               self.filter_size,
                                               curr_col_pic:curr_col_pic +
                                               self.filter_size]
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
                                   curr_row_pic:curr_row_pic + self.filter_size,
                                   curr_col_pic:curr_col_pic +
                                   self.filter_size] += (
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
        filter_size:
            Integer representing the size of the filter you will be
            using to scan the input images

        input_depth:
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
                 filter_size: int,
                 padding: Literal["valid", "same"] = "valid",
                 stride: int = 1,
                 poolType: Literal["max", "avg"] = "max",
                 final_conv_layer: bool = False):
        self.filter_size = filter_size
        self.type_pool = poolType
        self.padding = padding
        self.stride = stride
        self.final_conv_layer = final_conv_layer
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
            Numpy array of shape (numExamples, num_filters, self.filter_size,
            self.filter_size, self.num_filters) representing the spatially
            downsampled input
        """
        p = 0 if self.padding == "valid" else (self.filter_size - 1) / 2
        output_height = int((x.shape[2] - self.filter_size + 2 * p) /
                            (self.stride) + 1)
        output_width = int((x.shape[3] - self.filter_size + 2 * p) /
                           (self.stride) + 1)
        self.z = np.zeros((x.shape[0], x.shape[1], output_width, output_height))
        # technically it doesn't really make sense to pad in pool layers
        # but the option is still here
        padded_input = self._pad_input(x, self.filter_size, self.padding)
        self.ain = padded_input
        for i in range(padded_input.shape[0]):
            image = padded_input[i, :, :, :]
            # for every single image, you apply the spatial downsampling
            # PER dimension so we don't reduce the number of dimensions
            # we have, we just reduce the height and width
            for dimension in range(image.shape[0]):
                curr_row_pic = 0
                curr_row_neuron = -1
                # while you haven't seen all the height rows
                while curr_row_pic + self.filter_size <= image.shape[1]:
                    # while you haven't seen the full width of this row
                    curr_row_neuron += 1
                    # reset the column to zero for every single row we are at
                    # for both the picture and neuron
                    curr_col_pic = 0
                    curr_col_neuron = 0
                    while curr_col_pic + self.filter_size <= image.shape[2]:
                        curr_img_slice = image[dimension,
                                               curr_row_pic:curr_row_pic +
                                               self.filter_size,
                                               curr_col_pic:curr_col_pic +
                                               self.filter_size]
                        # max pool or average pool
                        if self.type_pool == "max":
                            self.z[i, dimension, curr_row_neuron,
                                   curr_col_neuron] = np.max(curr_img_slice)
                        else:
                            self.z[i, dimension, curr_row_neuron,
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
        if not self.final_conv_layer:
            return self.z
        else:
            return self.z.reshape(-1, x.shape[0])

    def update_weights(self,
                       dLdA: np.ndarray,
                       learn_rate: float,
                       epoch: int,
                       prediction_obj,
                       curr_x: np.ndarray,
                       curr_y: np.ndarray,
                       layer,
                       grad_check: bool = False):
        # Pooling layer has no weights to update but we still
        # need to get dL/d(A_layer-1) to pass back to the layer
        # before

        # if this is the last conv layer, then we reshape our output in
        # the forward pass to be a N features by M examples matrix

        # so when we're coming back, we have to reshape that to be
        # (num ex, num dim, H, W) IE same shape as the output of the
        # layer. Basically reverse of the flattening to a single vector
        # step :D
        if self.final_conv_layer:
            dLdA = dLdA.reshape(self.z.shape[0], self.z.shape[1],
                                self.z.shape[2], self.z.shape[3])

        # get dAout_dAin and chain it together with dLdA_out to get
        # dl_da_in to pass back
        dl_da_in = np.zeros((self.ain.shape[0], self.ain.shape[1],
                             self.ain.shape[2], self.ain.shape[3]))

        for i in range(self.ain.shape[0]):
            image = self.ain[i, :, :, :]
            for dimension in range(image.shape[0]):
                curr_row_pic = 0
                curr_row_neuron = -1
                while curr_row_pic + self.filter_size <= image.shape[1]:
                    curr_row_neuron += 1
                    curr_col_pic = 0
                    curr_col_neuron = 0
                    while curr_col_pic + self.filter_size <= image.shape[2]:
                        # this image slice is responsible for creating a
                        # SINGLE neuron. In other words, this is a multivariable
                        # scalar function as it takes multiple dimensions in
                        # and returns a single value
                        curr_img_slice = image[dimension,
                                               curr_row_pic:curr_row_pic +
                                               self.filter_size,
                                               curr_col_pic:curr_col_pic +
                                               self.filter_size]
                        # in max pool, dAout/dA_in is zero everywhere except the
                        # MAX idx
                        if self.type_pool == "max":
                            max_idx_row, max_idx_col = find_row_column_max_element(
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
                                     self.filter_size,
                                     curr_col_pic:curr_col_pic +
                                     self.filter_size] += (
                                         1 / np.size(curr_img_slice) *
                                         dLdA[i, dimension, curr_row_neuron,
                                              curr_col_neuron])

                        curr_col_pic += self.stride
                        curr_col_neuron += 1

                    curr_row_pic += self.stride

        return dl_da_in
