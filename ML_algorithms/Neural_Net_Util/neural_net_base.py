""" This module contains code representing the base class for Neural Networks
"""
import numpy as np
from ML_algorithms.Neural_Net_Util.neural_net_layers import DenseLayer, BatchNormLayer_Dense, BaseNeuralNetworkLayer
from ML_algorithms.Neural_Net_Util.ConvolutionalLayers import Conv2D
from ML_algorithms.Neural_Net_Util.Optimizers import gradientDescent, optimizer
from ML_algorithms.Neural_Net_Util.LossFunctions import LossFunction
from ML_algorithms.Neural_Net_Util.ActivationFunctions import Base_ActivationFunction
from ML_algorithms.Utility.misc import convertToHighestPred
from ML_algorithms.Utility.ScoreFunctions import accuracy
from typing import Union, Tuple, List
import copy


class NeuralNetworkBase(object):
    """
    This is a fully-connected Neural Network class, which can be used
    for supervised learning and unsupervised learning (autoencoders).

    Attributes:
        lossFunction:
            The objective to optimize during training

        input_features:
            Integer representing the number of input features a single
            example has in your dataset.

        layers (python list):
            List object containing all the layers present in the neural
            network
    """

    def __init__(self, lossFunction: LossFunction, input_features: int):
        self.layers = []
        self.lossFunction = lossFunction
        self.num_input = input_features

    def add_layer(self,
                  num_neurons: int,
                  activationFunction: Base_ActivationFunction,
                  isSoftmax: int = 0,
                  layer: BaseNeuralNetworkLayer = None,
                  keep_prob: int = 1) -> None:
        """
        This method adds a dense layer to your neural network.

        Args:
            num_neurons:
                Integer representing the number of neurons you would like in
                this dense layer of the neural network.

            activationFunction:
                Object of type Base_ActivationFunctionm which will be used to
                introduce non-linearity into your neural net

            isSoftmax:
                Integer that should be either 0 or 1 indicating whether the
                activation function is softmax or not

            layer:
                Object of type BaseNeuralNetworkLayer, indicating the current layer being
                added to the neural network. Default layer is a Dense layer.

        """
        if not self.layers:
            layer_x = DenseLayer(
                self.num_input, num_neurons, activationFunction,
                self.lossFunction.regularization,
                self.lossFunction.regParameter,
                isSoftmax) if layer is None else layer(
                    self.num_input, num_neurons, activationFunction,
                    self.lossFunction.regularization,
                    self.lossFunction.regParameter, isSoftmax, keep_prob)
            self.layers.append(layer_x)
        else:
            # if the layeer beforee this is a dense layer, then get its weight
            # shape otherwise if its a conv layer/ pool layer, we have no idea
            # how many neurons are going to be passed to this layer so set it
            # to None
            shape_1 = self.layers[-1].W.shape[0] if (
                isinstance(self.layers[-1], DenseLayer)
                and self.layers[-1].W is not None) else None
            layer_x = DenseLayer(shape_1, num_neurons, activationFunction,
                                 self.lossFunction.regularization,
                                 self.lossFunction.regParameter,
                                 isSoftmax) if layer is None else layer(
                                     shape_1, num_neurons, activationFunction,
                                     self.lossFunction.regularization,
                                     self.lossFunction.regParameter, isSoftmax,
                                     keep_prob)
            self.layers.append(layer_x)

    def fit(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        xvalid: Union[np.ndarray, None] = None,
        yvalid: Union[np.ndarray, None] = None,
        num_epochs: int = 10,
        batch_size: int = 32,
        ret_train_loss: bool = False,
        learn_rate: float = 0.1,
        optim: optimizer = gradientDescent(),
        verbose: bool = False
    ) -> Union[Tuple[int, int, int, int], Tuple[int, int], None]:
        """This method trains the neural network on the training set.

        M: number of training examples
        N: number of features in a single example

        Args:
            xtrain:
                Numpy matrix of shape (M, N) containing feature vectors

            ytrain:
                Numpy matrix of shape (M,1) containing the labels for the
                feature vectors

            xvalid:
                None or a numpy matrix of shape (M,N) containing the feature
                vectors used to validate the algorithm

            yvalid:
                None or a numpy vector of shape (M,1) containing the labels for
                xvalid

            num_epochs:
                Integer representing the number of epochs to train the model

            batch_size:
                Integer representing the number of examples to go through before
                performing a parameter update

            ret_train_loss:
                Boolean value indicating whether to return train loss and valid
                loss if validation set provided

            learn_rate:
                Floating point value indicating the learning rate to be used
                when optimizing the loss function

            optim:
                Object of type optimizer. Used to optimize the loss function

            verbose:
                Boolean value indicating whether to provide updates when
                training

        Returns:
            If ret_train_loss is set to True and xvalid is not None,
            4 integers will be returned in a tuple, indicating the training
            loss, validation loss, training accuracy, and validation accuracy
            respectively.

            If ret_train_loss is True and xvalid is None, then two integers
            will be returned in a tuple, indicating the training loss and
            training accuracy respectively.

            Otherwise, None will be returned.
        """
        # Dealing with edge case where you have less than 32 examples, which
        # can happen maybe for k-fold cv. Just do batch gradient descent if
        # the number of examples is super small
        if xtrain.shape[1] <= 1000 and len(xtrain.shape) == 2:
            batch_size = xtrain.shape[1]
            num_batches = 1
        # otherwise do mini batch gradient descent
        elif xtrain.shape[1] <= 1000 and len(xtrain.shape) == 2:
            num_batches = xtrain.shape[1] // batch_size
        else:
            num_batches = xtrain.shape[0] // batch_size
        train_loss = []
        train_acc = []
        validation_loss = []
        val_acc = []
        for epoch in range(num_epochs):
            currStart = 0
            currEnd = batch_size
            lossEpoch = []
            for i in range(num_batches):
                curr_y = ytrain[:, currStart:currEnd]
                if len(xtrain.shape) == 2:
                    curr_x = xtrain[:, currStart:currEnd]
                else:
                    # 3D pictures
                    curr_x = xtrain[currStart:currEnd, :, :, :]
                currStart = currEnd
                currEnd += batch_size
                pred_miniBatch = self._forward_propagate(curr_x)
                loss = self._calculateLoss(curr_y, pred_miniBatch, self.layers)
                lossEpoch.append(loss)
                backpropInit = self.lossFunction.derivativeLoss_wrtPrediction(
                    curr_y, pred_miniBatch)
                self._backward_propagate(backpropInit, learn_rate, optim, epoch,
                                         curr_x, curr_y)

            train_loss.append(np.mean(lossEpoch))

            if ytrain.shape[0] > 1:
                acc_trainSet = accuracy(convertToHighestPred(ytrain),
                                        self.predict(xtrain))
            else:
                acc_trainSet = accuracy(ytrain, self.predict(xtrain))
            train_acc.append(acc_trainSet)

            if xvalid is not None:
                if ytrain.shape[0] > 1:
                    acc_valSet = accuracy(convertToHighestPred(yvalid),
                                          self.predict(xvalid))
                    val_loss = self._calculateLoss(
                        yvalid, self._forward_propagate(xvalid), self.layers)
                else:
                    acc_valSet = accuracy(yvalid, self.predict(xvalid))
                    val_loss = self._calculateLoss(yvalid, self.predict(xvalid),
                                                   self.layers)

                val_acc.append(acc_valSet)
                validation_loss.append(val_loss)

            # provide updates during training for sanitys sake
            if verbose:
                print("Finished epoch %s" % (epoch))
                print("Train loss: %s, Train acc: %s" %
                      (train_loss[-1], train_acc[-1]))
                if xvalid is not None:
                    print("Valid loss: %s, Valid acc: %s" %
                          (validation_loss[-1], val_acc[-1]))

        if ret_train_loss and xvalid is not None:
            return train_loss, validation_loss, train_acc, val_acc
        elif ret_train_loss:
            return train_loss, train_acc

    def predict(self, X: np.ndarray, supervised: bool = True) -> np.ndarray:
        """
        This method is used to use the neural network to predict on instances
        it has not trained on.

        Args:
            X:
                Numpy matrix of shape (M,N) where M is the number of instances
                to predict on, and N is the number of features in an example.

            supervised:
                A boolean indicating whether this method is being used in a
                supervised neural network or an unsupervised neural network
                (ie: autoencoder).

        Returns:
            A numpy matrix containing the output of the neural network.
        """
        output = self._forward_propagate(X, train=False)
        # if more than one class, then compute the highest value as the
        # prediction
        if output.shape[0] > 1 and supervised:
            output = convertToHighestPred(output)
        return output

    def _convertToHighestPred(self, predictions):
        predictions = np.argmax(predictions, axis=0)
        return predictions

    def _calculateLoss(self, curr_y: np.ndarray, pred_minibatch: np.ndarray,
                       layersNet: List[BaseNeuralNetworkLayer]) -> float:
        """ This method is used to calculate the loss of the neural network
        on a batch of examples.

        M: Number of examples

        Args:
            curr_y:
                Numpy vector of shape (M,1) consisting of the labels for the
                M examples

            pred_minibatch:
                NumPy vector of shape (M,1) consisting of the predicted answers
                for the M examples

        Returns:
            Floating point value indicating the loss of the neural network on
            the given batch of examples
        """
        return self.lossFunction.get_loss(curr_y, pred_minibatch, layersNet)

    def _forward_propagate(self,
                           X: np.ndarray,
                           train: bool = True) -> np.ndarray:
        """ This method implements the forward propagation step for a neural
        network.

        Each layer is fed in the activations from the previous layer. a[L-1],
        is a matrix that will be of shape (M, N) where M is the number of
        training examples and N is the number of features, and computes its
        own activations a[L]. These activations are fed to the next layer
        and so on.

        Args:
            X:
                A NumPy matrix of shape (M,N) where M is the number of outputs
                and N is the number of features.

            train:
                A boolean indicating whether the algorithm is currently training
                or not

        Returns:
            A numpy matrix produced as output from the last hidden layer.
        """
        prev_activations = X
        for layer in self.layers:
            # if we are feeding in input from a Conv layer or pool layer
            # we don't know before how many activated neurons are going to
            # be passed into this dense layer, so we can't pre-initialize
            # the weights for each layer.
            if isinstance(layer, DenseLayer) and layer.W is None:
                # conv layer will have flattened its output to matrix shape
                layer.W, layer.b = layer._initializeWeights(
                    layer.num_layer, prev_activations.shape[0])
                if isinstance(layer, BatchNormLayer_Dense):
                    layer.gamma, layer.beta = layer._initializeGammaBeta()
                    layer.runningMean, layer.runningVar = layer._initializeRunningMeanVar(
                    )
            elif (isinstance(layer, Conv2D)) and layer.filters is None:
                layer.inputDepth = prev_activations.shape[1]
                layer.filters, layer.b = layer._initializeWeights()
            activations = layer.compute_forward(prev_activations, train)
            prev_activations = activations
        return activations

    def _backward_propagate(self, initalGradient: np.ndarray, learn_rate: float,
                            optim: optimizer, epoch: int, curr_x: np.ndarray,
                            curr_y: np.ndarray):
        """This method implements the backward propagation step
        for a neural network.

        The backpropagation is initialized by the gradient produced
        from the cost function dL/da. From there, we simply pass back
        through each of the layers in the neural network, with each
        layer computing dL/dZ, then from there getting dL/dW and dL/dB
        for this layer. The output from each layer will be dL/da[L-1],
        which is passed down further back in the circuit.

        Args:
            initialGradient:
                Numpy matrix representing starting gradient dL/da.

            learn_rate:
                Floating point value indicating the learning rate to
                be used when optimizing the cost function

            optim:
                Object of type optimizer to use to minimize the loss function

            epoch:
                Integer representing the epoch which we are training the network
                in currently

            curr_x:
                Numpy matrix representing the feature vectors currently being
                trained on

            curr_y:
                Numpy vector representing the labels for the feature vectors
        """
        dLdA = initalGradient
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if not layer.optim:
                layer.optim = copy.deepcopy(optim)
            dLdA_prev = layer._updateWeights(dLdA, learn_rate, epoch, self,
                                             curr_x, curr_y, i)
            dLdA = dLdA_prev
