""" This module contains code for functionality related to fully connected
neural network layers """
import numpy as np
import random
from typing import Literal, Union, Tuple
from ML_algorithms.Neural_Net_Util.ActivationFunctions import Base_ActivationFunction


class _BaseLayer(object):

    def compute_forward(self):
        raise NotImplementedError

    def updateWeights(self, dLdA):
        raise NotImplementedError

    def _getRegularizationLoss(self, regularizationType, regParameter,
                               numExamples, W):
        # Cost is averaged overall all examples so we get
        # Tot_cost_batch = 1/m * (loss_examples_batch + reg_loss_batch)
        # Tot_cost_batch = (1/m) * loss_examples_batch + (1/m)*reg_loss_batch
        # So each of dldW circuit and dldW regularization need to include the
        # 1/m term since it is included in the function we want to find the
        # derivative of
        if regularizationType == "L2":
            dregdW = (1 / numExamples) * regParameter * W
        else:
            signOfWeights = np.sign(W)
            dregdW = (1 / numExamples) * regParameter * signOfWeights
        return dregdW


class DenseLayer(_BaseLayer):
    """ This class represent a fully connected layer used in many machine
    learning algorithms, where every neuron has a connection to every neuron
    in the previous layer.

    M: The number of rows inside of a matrix, representing feature vectors

    Attributes:
        num_in:
            Integer representing the number of neurons that are expected
            as input to this layer

        num_layer:
            Integer representing the number of neurons in this layer

        activationFunction:
            Object of type Base_ActivationFunction representing the activation
            function to use within this layer

        W:
            Numpy array of shape (num_layer, num_in) of weights for this layer

        b:
            Numpy array of shape (num_layer, 1) representing the bias values
            for the neurons in this layer

        Z:
            Numpy array of shape (num_layer, M) consisting of linear sums for
            all input examples

        A:
            Numpy array of shape (num_layer, M) representing the matrix Z after
            it has been fed into an activation function

        Ain:
            Numpy array of shape (num_layer-1, M) representing the input to this
            layer

        regParameter:
            Floating point value representing the strength of the regularization
    """

    def __init__(self,
                 num_in: int,
                 num_layer: int,
                 activationFunction: Base_ActivationFunction,
                 regularization=None,
                 regParameter: float = None,
                 isSoftmax: Literal[0, 1] = 0,
                 keepProb: float = None):
        self.num_in = num_in
        self.num_layer = num_layer
        self.W, self.b = self._initializeWeights(self.num_layer, self.num_in)
        self.Z = None
        self.A = None
        self.Ain = None
        self.activationFunction = activationFunction
        self.regularization = regularization
        self.regParameter = regParameter
        self.isSoftmax = isSoftmax
        self.optim = None

    def _initializeWeights(
        self, num_layer: int, num_prev_layer: int
    ) -> Union[Tuple[None, None], Tuple[np.ndarray, np.ndarray]]:
        """ This method will initialize the weights used in this layer. For the
        bias vector, a simple zeros initialization will be used. For the weight
        matrix, the values will be sampled from a standard normal distribution
        and then multiplied by 0.01 to prevent the issue of exploding and
        vanishing gradients.

        Args:
            num_layer:
                Integer representing the number of neurons present in the
                current dense layer

            num_prev_layer:
                Integer representing the number of neurons present in the
                previous dense layer
        """
        if num_prev_layer is None:
            return None, None
        else:
            W = np.random.randn(num_layer, num_prev_layer) * 0.01
            b = np.zeros((num_layer, 1))
            return W, b

    def compute_forward(self,
                        prevlayer_activations: np.ndarray,
                        train: bool = True) -> np.ndarray:
        """ This method computes the forward pass through this layer.

        Args:
            prevlayer_activations:
                A (nl_prev, M) numpy array containing the activations for all M
                examples for every neuron in the previous layer, where nl_prev
                is the number of neurons in the previous layer.

            train:
                Boolean value indicating whether the layer is currently training

        Returns:
            A numpy matrix of shape (self.num_layer, M) where self.num_layer
            is the number of neurons inside of this layer, and M is the number
            of examples in the input matrix
        """
        assert self.W.shape[1] == prevlayer_activations.shape[
            0], "Your weights and inputs shapes are mismatched!"
        self.Ain = prevlayer_activations
        self.Z = np.dot(self.W, prevlayer_activations) + self.b
        self.A = self.activationFunction.compute_output(self.Z)
        return self.A

    def _updateWeights(self,
                       dLdA,
                       learn_rate,
                       epoch,
                       prediction_obj,
                       curr_x,
                       curr_y,
                       layer,
                       gradCheck=False):
        """
        This method computes the backward pass through this layer. In the gradient circuit,
        this layers job is to recieve the Jacobian matrix of the loss function with respect
        to this layers activations, and update its parameters, and then pass on the gradient of the 
        loss function with respect to the previous layers activations. 


        Parameters:
        - dLdA (NumPy matrix) -> A (nl, M) NumPy matrix containing the 
        Jacobian matrix of the loss function with respect to the activations 
        in this layer.

        - optim (function) -> optimizer to use to minimize the loss function 

        - learn_rate (float) -> learning rate to be used when optimizing the cost function

        - epoch (int) -> the epoch we are updating for currently
        
        - prediction_obj (NeuralNet obj) -> the base neural network object we can use

        - curr_x (NumPy matrix) -> matrix of examples we are currently training on

        - curr_y (NumPy vector) -> matrix of labels for the examples

        - layer (int) -> layer in the network we are currently updating 

        Returns: dL/dA_L-1 (NumPy Matrix) -> A (nl-1, M) NumPy matrix containing the Jacobian Matrix
        of the loss function with respect to the activations in the previous layer.
        """

        ## GRADIENT GOING BACKWARDS IN CIRCUIT
        if self.isSoftmax == 0:
            dadz = self.activationFunction.getDerivative_wrtInput(self.Z)
            dLdZ = dLdA * dadz
        else:
            # Every example produces its own jacobian matrix for the softmax function
            # Therefore you have to loop through every example, get the Jacobian matrix for that example
            # and dot product it with the dL/da for this example
            # You can combine the gradients of the softmax and the cross entropy and simplify to make more efficient
            # But this is more explicit of what is actually happening for learning purposes
            dLdZ = dLdZ_sm(self.Z, self.A, dLdA, self.activationFunction)

        dLdW = np.dot(dLdZ, self.Ain.T)
        dLdB = np.sum(dLdZ, axis=1, keepdims=True)
        dLdA_prevLayer = np.dot(self.W.T, dLdZ)

        assert dLdW.shape == self.W.shape, "Your W[L] shape is not the same as dW/dW[L] shape"
        assert dLdB.shape == self.b.shape, "Your B[L] shape is not the same as dW/dB[L] shape"
        assert dLdA_prevLayer.shape == self.Ain.shape, "Your dL/dA[L-1] shapes are incomptabile"

        # Epoch zero and you want to gradient check, do some gradient checks for params W and b
        if epoch == 0 and gradCheck:
            self._gradientCheck("W", dLdW, curr_x, curr_y, prediction_obj,
                                layer)
            self._gradientCheck("b", dLdB, curr_x, curr_y, prediction_obj,
                                layer)

        ## GRADIENT FROM REGULARIZATION IF REGULARIZATION

        dregdW = 0
        if self.regularization != None:
            dregdW = self._getRegularizationLoss(self.regularization,
                                                 self.regParameter,
                                                 self.Ain.shape[1], self.W)
        dLdW = dLdW + dregdW
        self.W, self.b = self.optim.updateParams([self.W, self.b], [dLdW, dLdB],
                                                 learn_rate,
                                                 epochNum=epoch + 1)
        return dLdA_prevLayer

    def _gradientCheck(self, param, dparam, x, y, obj, layer, num_checks=10):
        """
        This method checks the gradient we are using to update the params of this
        dense layer. This check is quite expensive, so its only done on the first 
        epoch. 

        Parameters:
        - param (string) -> the parameter currently being checked
        - dparam (NumPy matrix) -> the dL/dparam in question
        - x (NumPy Matrix) -> the matrix of examples used to calculate the gradient for
        - y (NumPy Matrix) -> the labels for the examples
        - obj (NeuralNet Object) -> object so we have access to forward prop step
        - layer (int) -> the specific layer in the object we are gradient checking for 
        - num_checks (int) -> the number of different dimensions to check in the parameter matrix

        Returns: None 
        """
        eps = 1e-5
        random.seed(21)
        # Get access to the objects params - W or B that we are using to predict with
        obj_attr = getattr(obj.layers[layer], param)
        for i in range(num_checks):
            changeIdx = random.randrange(0, len(obj_attr.T))
            savedParam = obj_attr[:, changeIdx]

            # Update the param by eps, then decrease param by eps, then
            # calculate the new loss in both cases so we can get dL/dparam and compare to
            # analytical gradient
            obj_attr[:, changeIdx] += eps
            preds1 = obj._forward_propagate(x)
            loss_higher = obj._calculateLoss(y, preds1)

            obj_attr[:, changeIdx] = savedParam - eps
            preds2 = obj._forward_propagate(x)
            loss_lower = obj._calculateLoss(y, preds2)

            obj_attr[:, changeIdx] = savedParam
            grad_analytic = dparam[:, changeIdx]
            grad_numeric = (loss_higher - loss_lower) / (2 * eps)
            rel_error = abs(grad_analytic - grad_numeric) / abs(grad_analytic +
                                                                grad_numeric)


def dLdZ_sm(Z, A, dLdA, activFunc, efficient=0):
    """
    This function hooks up dL/dA with dA/dZ to produce the dL/dZ through the softmax layer.
    
    We have to do a for loop to get the jacobian matrix of the softmax for every single input example,
    which we insert into the overall dL/dZ for the layer. 

    The output of this function is equivalent to dividing the activations by m if the label
    for the current class is 0, and if the true label is y= 1 at a certain activation, 
    then it is equal to (activation/m)-1. 

    Parameters:
    -> Z (NumPy matrix): NumPy matrix of shape (C, M) containing the raw logits for the softmax layer 
    -> A (NumPy matrix): NumPy matrix of shape (C,M) containing the activations for the softmax layer
    -> dLdA (NumPy matrix): NumPy matrix of shape (C,M) containing the derivative of the loss function
    -> activFunc (object): Activation Function object used to get dLdZ_sm 
    """
    if not efficient:
        dLdZ = np.zeros((Z.shape[0], Z.shape[1]))
        for i in range(A.shape[1]):
            column_vecActiv = A[:, i].reshape(-1, 1)
            derivsLoss_ithEx = dLdA[:, i].reshape(-1, 1)
            jacobianActiv = activFunc.getDerivative_wrtInput(column_vecActiv)
            assert jacobianActiv.shape[1] == derivsLoss_ithEx.shape[0]
            dLdZ_ithEx = np.dot(jacobianActiv, derivsLoss_ithEx)
            assert dLdZ_ithEx.shape == (Z.shape[0], 1)
            dLdZ[:, i] = dLdZ_ithEx.reshape(-1)
        return dLdZ


class BatchNormLayer_Dense(DenseLayer):

    def __init__(self,
                 num_in,
                 num_layer,
                 activationFunction,
                 regularization=None,
                 regParameter=None,
                 isSoftmax=0,
                 p=None):
        super(BatchNormLayer_Dense,
              self).__init__(num_in, num_layer, activationFunction,
                             regularization, regParameter, isSoftmax)
        # these are learnable parameters
        self.gamma, self.beta = self._initializeGammaBeta()
        # We need to keep an exponentially weighted average of the mean and variance
        # for this layer when we train, so we can use this to normalize test time predictions
        # init to a vector of zeros - one mean and one variance for every neuron in this layer
        self.runningMean, self.runningVar = self._initializeRunningMeanVar()
        self.Z_in = None
        self.Z_centered = None
        self.Z_norm = None
        self.Z_final = None
        # for backprop you have to cache stuff for the current pass
        self.variance = None
        self.inv_stdDev = None
        self.mean_miniBatch = None
        self.eps = 1e-7

    def _initializeGammaBeta(self):
        # initialize gamma to be a vector of 1's for every neuron in this layer
        # and beta to be 0 for every neuron in this layer
        if self.W is None:
            return None, None
        else:
            return np.ones((self.W.shape[0], 1)), np.zeros((self.W.shape[0], 1))

    def _initializeRunningMeanVar(self):
        if self.W is None:
            return None, None
        else:
            return np.zeros((self.W.shape[0], 1)), np.zeros(
                (self.W.shape[0], 1))

    def compute_forward(self, x, train=True):
        if train:
            return self._trainForward(x)
        return self._testForward(x)

    def _trainForward(self, prevlayer_activations):
        assert self.W.shape[1] == prevlayer_activations.shape[
            0], "Your weights and inputs shapes are mismatched!"
        self.Ain = prevlayer_activations
        self.Z_in = np.dot(self.W, prevlayer_activations) + self.b
        # Z is a (features, examples) matrix so for every feature, we want the mean and std dev
        # of every row meaning we get a (N,1) vector of means
        self.mean_miniBatch = np.mean(self.Z_in, axis=1, keepdims=True)
        self.variance = np.var(self.Z_in, axis=1, keepdims=True)
        # update running mean and std dev of every feature
        self._updateRunningAvg()
        # normalize feature means to 0 and std dev to 1
        self.Z_centered = self.Z_in - self.mean_miniBatch
        self.inv_stdDev = 1 / np.sqrt(self.variance + self.eps)
        self.Z_norm = self.Z_centered * self.inv_stdDev
        # Multiply by learnable parameters to avoid the network from losing expressivity
        self.Z_final = self.gamma * self.Z_norm + self.beta
        # Finally, feed into activation function to activate.
        self.A = self.activationFunction.compute_output(self.Z_final)
        return self.A

    def _updateRunningAvg(self, beta=0.9):
        self.runningMean = (beta) * (self.runningMean) + (
            1 - beta) * self.mean_miniBatch
        self.runningVar = (beta) * (self.runningVar) + (1 -
                                                        beta) * self.variance

    def _testForward(self, prevlayer_activations):
        # BatchNorm has diff behaviour at test time and train time
        assert self.W.shape[1] == prevlayer_activations.shape[
            0], "Your weights and inputs shapes are mismatched!"
        Z = np.dot(self.W, prevlayer_activations) + self.b
        # Vectorize operation - elementwise subtraction, then elementwise division
        # where Z is a (N,M) matrix and running mean and running variance are (N,1) vectors
        normalized_Z = (Z - self.runningMean) / (np.sqrt(self.runningVar +
                                                         self.eps))
        # multiply by learnable parameter gamma and add beta
        z_final = self.gamma * normalized_Z + self.beta
        return self.activationFunction.compute_output(z_final)

    def _updateWeights(self,
                       dLdA,
                       learn_rate,
                       epoch,
                       prediction_obj,
                       curr_x,
                       curr_y,
                       layer,
                       gradCheck=False):
        """
        This method computes the backward pass through this layer. In the gradient circuit,
        this layers job is to recieve the Jacobian matrix of the loss function with respect
        to this layers activations, and update its parameters, and then pass on the gradient of the 
        loss function with respect to the previous layers activations. 

        Parameters:
        -> dLdA (NumPy matrix) -> A (nl, M) NumPy matrix containing the 
        Jacobian matrix of the loss function with respect to the activations 
        in this layer.

        -> optim (function) -> optimizer to use to minimize the loss function 

        -> learn_rate (float) -> learning rate to be used when optimizing the cost function

        -> epoch (int) -> the epoch we are updating for currently
        
        -> prediction_obj (NeuralNet obj) -> the base neural network object we can use

        -> curr_x (NumPy matrix) -> matrix of examples we are currently training on

        -> curr_y (NumPy vector) -> matrix of labels for the examples

        -> layer (int) -> layer in the network we are currently updating 

        Returns: dL/dA_L-1 (NumPy Matrix) -> A (nl-1, M) NumPy matrix containing the Jacobian Matrix
        of the loss function with respect to the activations in the previous layer.
        """
        ## GRADIENT GOING BACKWARDS IN CIRCUIT
        if self.isSoftmax == 0:
            dadz_final = self.activationFunction.getDerivative_wrtInput(
                self.Z_final)
            dLdZ_final = dLdA * dadz_final
        else:
            dLdZ_final = dLdZ_sm(self.Z_final, self.A, dLdA,
                                 self.activationFunction)

        # gradients for learnable parameters - computation step for z_final layer
        dLdGamma = np.sum(np.dot(dLdZ_final, self.Z_norm.T),
                          axis=1,
                          keepdims=True)
        dLdBeta = np.sum(dLdZ_final, axis=1, keepdims=True)
        dLdZnorm = dLdZ_final * self.gamma

        # first branch dLdZ
        dLdZin_firstBranch = dLdZnorm * self.inv_stdDev
        ## variance portion of dLdZ
        dZnormdInv = self.Z_centered
        dInvdVar = -(0.5) * np.power(self.variance + self.eps, -3 / 2)
        dLdVar = np.sum(dLdZnorm * dZnormdInv * dInvdVar, axis=1, keepdims=True)
        dLdZin_secondBranch = dLdVar * (2 /
                                        self.Z_in.shape[1]) * (self.Z_centered)
        ## mean portion of dLdZ
        dLdMu1 = np.sum(dLdZnorm * (-1) * (self.inv_stdDev),
                        axis=1,
                        keepdims=True)
        dLdMu2 = np.sum(dLdVar * (-2 / self.Z_in.shape[1]) * (self.Z_centered),
                        axis=1,
                        keepdims=True)
        dLdZin_thirdBranch = (1 / self.Z_in.shape[1]) * (dLdMu1 + dLdMu2)
        # total dLdZ is sum of all three branches
        dLdZ_in = dLdZin_firstBranch + dLdZin_secondBranch + dLdZin_thirdBranch

        # finally get back to the un-normalized activations where we can get what we
        # wanted from the beginning
        dLdW = np.dot(dLdZ_in, self.Ain.T)
        dLdB = np.sum(dLdZ_in, axis=1, keepdims=True)
        dLdA_prevLayer = np.dot(self.W.T, dLdZ_in)

        assert dLdW.shape == self.W.shape, "Your W[L] shape is not the same as dW/dW[L] shape"
        assert dLdB.shape == self.b.shape, "Your B[L] shape is not the same as dW/dB[L] shape"
        assert dLdA_prevLayer.shape == self.Ain.shape, "Your dL/dA[L-1] shapes are incomptabile"
        assert dLdGamma.shape == self.gamma.shape, "Your dL/dGamma shape is not the same as gammas shape"
        assert dLdBeta.shape == self.beta.shape, "Your dL/dBeta shape is not the same as betas shape"

        # Epoch zero and you want to gradient check, do some gradient checks for params W and b
        if epoch == 0 and gradCheck:
            self._gradientCheck("W", dLdW, curr_x, curr_y, prediction_obj,
                                layer)
            self._gradientCheck("b", dLdB, curr_x, curr_y, prediction_obj,
                                layer)

        ## GRADIENT FROM REGULARIZATION IF REGULARIZATION
        dregdW = 0
        if self.regularization != None:
            dregdW = self._getRegularizationLoss(self.regularization,
                                                 self.regParameter,
                                                 self.Ain.shape[1], self.W)
        dLdW = dLdW + dregdW
        self.W, self.b, self.gamma, self.beta = self.optim.updateParams(
            [self.W, self.b, self.gamma, self.beta],
            [dLdW, dLdB, dLdGamma, dLdBeta],
            learn_rate,
            epochNum=epoch + 1)
        return dLdA_prevLayer


class DropOutLayer_Dense(DenseLayer):

    def __init__(self,
                 num_in,
                 num_layer,
                 activationFunction,
                 regularization=None,
                 regParameter=None,
                 isSoftmax=0,
                 keepProb=0.5):
        super(DropOutLayer_Dense,
              self).__init__(num_in, num_layer, activationFunction,
                             regularization, regParameter, isSoftmax)
        self.keepProb = keepProb
        # if keep prob is 1, then every value in matrix will be less than 1, and thus be True and so every
        # activated neuron is kept in. If keepProb is 0, then every neuron is dropped.

    def compute_forward(self, x, train=True):
        if train:
            return self._trainForward(x)
        return self._testForward(x)

    def _trainForward(self, prevlayer_activations):
        assert self.W.shape[1] == prevlayer_activations.shape[
            0], "Your weights and inputs shapes are mismatched!"
        self.Ain = prevlayer_activations
        self.Z = np.dot(self.W, prevlayer_activations) + self.b
        self.mask = (np.random.rand(*self.Z.shape) < self.keepProb).astype(int)
        A_undropped = self.activationFunction.compute_output(self.Z)
        A_dropped = A_undropped * self.mask
        # have to divide by keepProb because we don't want the activations of the next layer
        # to have lower values just cause we're feeding in so many values that are zero
        self.A = A_dropped / self.keepProb
        return self.A

    def _testForward(self, prevlayer_activations):
        # testing forward has no dropout - we use our full architecture
        # so its just the same as forward prop through a dense layer
        assert self.W.shape[1] == prevlayer_activations.shape[
            0], "Your weights and inputs shapes are mismatched!"
        Ain = prevlayer_activations
        Z = np.dot(self.W, prevlayer_activations) + self.b
        A = self.activationFunction.compute_output(self.Z)
        return A

    def _updateWeights(self,
                       dLdA,
                       learn_rate,
                       epoch,
                       prediction_obj,
                       curr_x,
                       curr_y,
                       layer,
                       gradCheck=False):
        """
        This method computes the backward pass through this layer. In the gradient circuit,
        this layers job is to recieve the Jacobian matrix of the loss function with respect
        to this layers activations, and update its parameters, and then pass on the gradient of the 
        loss function with respect to the previous layers activations. 


        Parameters:
        -> dLdA (NumPy matrix) -> A (nl, M) NumPy matrix containing the 
        Jacobian matrix of the loss function with respect to the activations 
        in this layer.

        -> optim (function) -> optimizer to use to minimize the loss function 

        -> learn_rate (float) -> learning rate to be used when optimizing the cost function

        -> epoch (int) -> the epoch we are updating for currently
        
        -> prediction_obj (NeuralNet obj) -> the base neural network object we can use

        -> curr_x (NumPy matrix) -> matrix of examples we are currently training on

        -> curr_y (NumPy vector) -> matrix of labels for the examples

        -> layer (int) -> layer in the network we are currently updating 

        Returns: dL/dA_L-1 (NumPy Matrix) -> A (nl-1, M) NumPy matrix containing the Jacobian Matrix
        of the loss function with respect to the activations in the previous layer.
        """

        ## GRADIENT GOING BACKWARDS IN CIRCUIT
        # you dont use a dropout layer w/ a softmax activation

        # only change between dropout and normal dense layer - you have the boolean mask and /p term
        # when getting dadz
        dadz = (self.activationFunction.getDerivative_wrtInput(self.Z) *
                self.mask) / (self.keepProb)
        dLdZ = dLdA * dadz

        dLdW = np.dot(dLdZ, self.Ain.T)
        dLdB = np.sum(dLdZ, axis=1, keepdims=True)
        dLdA_prevLayer = np.dot(self.W.T, dLdZ)

        assert dLdW.shape == self.W.shape, "Your W[L] shape is not the same as dW/dW[L] shape"
        assert dLdB.shape == self.b.shape, "Your B[L] shape is not the same as dW/dB[L] shape"
        assert dLdA_prevLayer.shape == self.Ain.shape, "Your dL/dA[L-1] shapes are incomptabile"

        # Epoch zero and you want to gradient check, do some gradient checks for params W and b
        if epoch == 0 and gradCheck:
            self._gradientCheck("W", dLdW, curr_x, curr_y, prediction_obj,
                                layer)
            self._gradientCheck("b", dLdB, curr_x, curr_y, prediction_obj,
                                layer)

        ## GRADIENT FROM REGULARIZATION IF REGULARIZATION
        dregdW = 0
        if self.regularization != None:
            dregdW = self._getRegularizationLoss(self.regularization,
                                                 self.regParameter,
                                                 self.Ain.shape[1], self.W)
        dLdW = dLdW + dregdW
        self.W, self.b = self.optim.updateParams([self.W, self.b], [dLdW, dLdB],
                                                 learn_rate,
                                                 epochNum=epoch + 1)
        return dLdA_prevLayer
