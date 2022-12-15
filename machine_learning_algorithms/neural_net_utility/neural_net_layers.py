""" This module contains code for functionality related to fully connected
neural network layers """
import numpy as np
import random
from typing import Literal, Union, Tuple, TYPE_CHECKING
from machine_learning_algorithms.neural_net_utility.activation_functions import BaseActivationFunction
if TYPE_CHECKING:
    from machine_learning_algorithms.neural_net_utility.neural_net_base import NeuralNetworkBase


class BaseNeuralNetworkLayer:

    def __init__(self,
                 num_in: int,
                 num_layer: int,
                 activation_function: BaseActivationFunction,
                 regularization: Union[Literal["L1", "L2"], None] = None,
                 reg_parameter: Union[float, None] = None,
                 isSoftmax: Literal[0, 1] = 0,
                 keepProb: Union[float, None] = None):
        self.num_in = num_in
        self.num_layer = num_layer
        self.W, self.b = self._initialize_weights(self.num_layer, self.num_in)
        self.Z = None
        self.A = None
        self.Ain = None
        self.activation_function = activation_function
        self.regularization = regularization
        self.reg_parameter = reg_parameter
        self.isSoftmax = isSoftmax
        self.optim = None

    def _initialize_weights(
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
            w = np.random.randn(num_layer, num_prev_layer) * 0.01
            b = np.zeros((num_layer, 1))
            return w, b

    def compute_forward(self):
        raise NotImplementedError

    def update_weights(self, dl_da):
        raise NotImplementedError

    def _getRegularizationLoss(self, regularizationType, reg_parameter,
                               numExamples, W):
        # Cost is averaged overall all examples so we get
        # Tot_cost_batch = 1/m * (loss_examples_batch + reg_loss_batch)
        # Tot_cost_batch = (1/m) * loss_examples_batch + (1/m)*reg_loss_batch
        # So each of dldW circuit and dldW regularization need to include the
        # 1/m term since it is included in the function we want to find the
        # derivative of
        if regularizationType == "L2":
            dreg_dw = (1 / numExamples) * reg_parameter * W
        else:
            sign_of_weights = np.sign(W)
            dreg_dw = (1 / numExamples) * reg_parameter * sign_of_weights
        return dreg_dw


class DenseLayer(BaseNeuralNetworkLayer):
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

        activation_function:
            Object of type BaseActivationFunction representing the activation
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

        reg_parameter:
            Floating point value representing the strength of the regularization
    """

    def compute_forward(self,
                        prevlayer_activations: np.ndarray,
                        train: bool = True) -> np.ndarray:
        """ This method computes the forward pass through this layer.

        nl_prev: Number of neurons in the previous layer
        M: Number of examples in the input

        Args:
            prevlayer_activations:
                A (nl_prev, M) numpy array containing the activations for all M
                examples for every neuron in the previous layer

            train:
                Boolean value indicating whether the layer is currently training

        Returns:
            A numpy matrix of shape (self.num_layer, M) where self.num_layer
            is the number of neurons inside of this layer
        """
        assert self.W.shape[1] == prevlayer_activations.shape[
            0], "Your weights and inputs shapes are mismatched!"
        self.Ain = prevlayer_activations
        self.Z = np.dot(self.W, prevlayer_activations) + self.b
        self.A = self.activation_function.compute_output(self.Z)
        return self.A

    def update_weights(self,
                       dLdA: np.ndarray,
                       learn_rate: float,
                       epoch: int,
                       prediction_obj: "NeuralNetworkBase",
                       curr_x: np.ndarray,
                       curr_y: np.ndarray,
                       layer: int,
                       grad_check: bool = False) -> np.ndarray:
        """ This method computes the backward pass through this layer. In
        the gradient circuit, this layers job is to recieve the jacobian
        matrix of the loss function with respect to this layers activations,
        and update its parameters, and then pass on the gradient of the
        loss function with respect to the previous layers activations.

        nl: Number of neurons in the current layer
        M: number of examples
        N: Number of features in the input

        Args:
            dLdA:
                A (nl, M) numpy array containing the jacobian matrix of the loss
                function with respect to the activations in this layer.

            optim:
                An object of type Optimizer, used to minimize the loss function

            learn_rate:
                A floating point value representing the learning rate to be used
                when optimizing the cost function

            epoch:
                An integer representing the epoch we are updating for currently

            prediction_obj:
                Object representing the base neural network object we can use

            curr_x:
                A numpy array of shape (N,M), representing the examples that the
                layer is currently training on

            curr_y:
                A numpy array of shape (1,M), representing the labels for the
                examples

            layer:
                An integer representing the layer in the network we are
                currently updating

            grad_check:
                Variable of type boolean representing whether we are performing
                gradient checking in the layer

        Returns:
            A (nl-1, M) numpy array containing the jacobian matrix of the loss
            function with respect to the activations in the previous layer.
        """

        ## GRADIENT GOING BACKWARDS IN CIRCUIT
        if self.isSoftmax == 0:
            dadz = self.activation_function.get_derivative_wrt_input(self.Z)
            dl_dz = dLdA * dadz
        else:
            # Every example produces its own jacobian matrix for the softmax
            # function. Therefore you have to loop through every example, get
            # the jacobian matrix for that example and dot product it with the
            # dL/da for this example. You can combine the gradients of the
            # softmax and the cross entropy and simplify to make more efficient.
            # But this is more explicit of what is actually happening for
            # learning purposes
            dl_dz = dl_dz_softmax(self.Z, self.A, dLdA,
                                  self.activation_function)

        dl_dw = np.dot(dl_dz, self.Ain.T)
        dl_db = np.sum(dl_dz, axis=1, keepdims=True)
        dl_da_prev_layer = np.dot(self.W.T, dl_dz)

        assert dl_dw.shape == self.W.shape, (
            "Your W[L] shape is not the same as dW/dW[L] shape")
        assert dl_db.shape == self.b.shape, (
            "Your B[L] shape is not the same as dW/dB[L] shape")
        assert dl_da_prev_layer.shape == self.Ain.shape, (
            "Your dL/dA[L-1] shapes are incomptabile")

        # Epoch zero and you want to gradient check, do some gradient checks
        # for params W and b
        if epoch == 0 and grad_check:
            self._gradient_check("W", dl_dw, curr_x, curr_y, prediction_obj,
                                 layer)
            self._gradient_check("b", dl_db, curr_x, curr_y, prediction_obj,
                                 layer)

        ## GRADIENT FROM REGULARIZATION IF REGULARIZATION

        dreg_dw = 0
        if self.regularization is not None:
            dreg_dw = self._getRegularizationLoss(self.regularization,
                                                  self.reg_parameter,
                                                  self.Ain.shape[1], self.W)
        dl_dw = dl_dw + dreg_dw
        self.W, self.b = self.optim.update_params([self.W, self.b],
                                                  [dl_dw, dl_db],
                                                  learn_rate,
                                                  epoch_num=epoch + 1)
        return dl_da_prev_layer

    def _gradient_check(self,
                        param: str,
                        dparam: np.ndarray,
                        x: np.ndarray,
                        y: np.ndarray,
                        obj: "NeuralNetworkBase",
                        layer: int,
                        num_checks: int = 10) -> None:
        """ This method checks the gradient we are using to update the params
        of this dense layer. This check is quite expensive, so its only done
        on the first epoch.

        Args:
            param:
                String representing the parameter currently being checked

            dparam:
                Numpy array representing the dL/dparam in question

            x:
                Numpy array representing the matrix of examples used to
                calculate the gradient for

            y:
                The labels for the examples

            obj:
                Neural network object so we have access to forward prop step

            layer:
                Integer representing the specific layer in the object we are
                gradient checking for

            num_checks:
                Integer representing the number of different dimensions to check
                in the parameter matrix
        """
        eps = 1e-5
        random.seed(21)
        # Get access to the objects params - W or B that we are using to
        # predict with
        obj_attr = getattr(obj.layers[layer], param)
        for _ in range(num_checks):
            change_idx = random.randrange(0, len(obj_attr.T))
            saved_param = obj_attr[:, change_idx]

            # Update the param by eps, then decrease param by eps, then
            # calculate the new loss in both cases so we can get dL/dparam
            # and compare to analytical gradient
            obj_attr[:, change_idx] += eps
            preds1 = obj._forward_propagate(x)
            loss_higher = obj._calculateLoss(y, preds1)

            obj_attr[:, change_idx] = saved_param - eps
            preds2 = obj._forward_propagate(x)
            loss_lower = obj._calculateLoss(y, preds2)

            obj_attr[:, change_idx] = saved_param
            grad_analytic = dparam[:, change_idx]
            grad_numeric = (loss_higher - loss_lower) / (2 * eps)
            rel_error = abs(grad_analytic - grad_numeric) / abs(grad_analytic +
                                                                grad_numeric)


def dl_dz_softmax(z: np.ndarray,
                  a: np.ndarray,
                  dl_da: np.ndarray,
                  activ_func: BaseActivationFunction,
                  efficient: bool = False) -> Union[None, np.ndarray]:
    """ This function hooks up dL/dA with dA/dZ to produce the
    dL/dZ through the softmax layer.

    We have to do a for loop to get the jacobian matrix of the
    softmax for every single input example, which we insert into
    the overall dL/dZ for the layer.

    The output of this function is equivalent to dividing the
    activations by m if the label for the current class is 0,
    and if the true label is y= 1 at a certain activation,
    then it is equal to (activation/m)-1.

    C: Number of neurons in the softmax layer
    M: Number of examples

    Args:
        z:
            Numpy array of shape (C, M) containing the raw logits
            for the softmax layer

        a:
            Numpy array of shape (C, M) containing the activations for
            the softmax layer

        dl_da:
            Numpy array of shape (C, M) containing the derivative of the
            loss function

        activ_func:
            Activation Function object used to get dl_dz_softmax

        efficient:
            Boolean indicating whether to get the gradient for the softmax
            layer the efficient way

    Returns:
        None if the gradient is to be computed efficiently. Otherwise, a
        numpy array of shape (C, M) will be returned which represents the
        jacobian matrix dl_dz for the softmax layer.
    """
    if not efficient:
        dl_dz = np.zeros((z.shape[0], z.shape[1]))
        for i in range(a.shape[1]):
            column_vectors_activated = a[:, i].reshape(-1, 1)
            deriv_loss_ith_example = dl_da[:, i].reshape(-1, 1)
            jacobian_activation = activ_func.get_derivative_wrt_input(
                column_vectors_activated)
            assert jacobian_activation.shape[1] == deriv_loss_ith_example.shape[
                0]
            dl_dz_ith_ex = np.dot(jacobian_activation, deriv_loss_ith_example)
            assert dl_dz_ith_ex.shape == (z.shape[0], 1)
            dl_dz[:, i] = dl_dz_ith_ex.reshape(-1)
        return dl_dz


class DenseBatchNormLayer(DenseLayer):

    def __init__(self,
                 num_in: int,
                 num_layer: int,
                 activation_function,
                 regularization=None,
                 reg_parameter: Union[float, None] = None,
                 isSoftmax: Literal[0, 1] = 0,
                 p=None):
        super(DenseBatchNormLayer,
              self).__init__(num_in, num_layer, activation_function,
                             regularization, reg_parameter, isSoftmax)
        # these are learnable parameters
        self.gamma, self.beta = self._init_gamma_beta()
        # We need to keep an exponentially weighted average of the
        # mean and variance for this layer when we train, so we can
        # use this to normalize test time predictions
        self.runningMean, self.runningVar = self._init_running_mean_var()
        self.Z_in = None
        self.Z_centered = None
        self.Z_norm = None
        self.Z_final = None
        # for backprop you have to cache stuff for the current pass
        self.variance = None
        self.inv_stdDev = None
        self.mean_miniBatch = None
        self.eps = 1e-7

    def _init_gamma_beta(self):
        # initialize gamma to be a vector of 1's for every neuron in this layer
        # and beta to be 0 for every neuron in this layer
        if self.W is None:
            return None, None
        else:
            return np.ones((self.W.shape[0], 1)), np.zeros((self.W.shape[0], 1))

    def _init_running_mean_var(self):
        if self.W is None:
            return None, None
        else:
            return np.zeros((self.W.shape[0], 1)), np.zeros(
                (self.W.shape[0], 1))

    def compute_forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        if train:
            return self._train_forward(x)
        return self._test_forward(x)

    def _train_forward(self, prevlayer_activations: np.ndarray) -> np.ndarray:
        assert self.W.shape[1] == prevlayer_activations.shape[
            0], "Your weights and inputs shapes are mismatched!"
        self.Ain = prevlayer_activations
        self.Z_in = np.dot(self.W, prevlayer_activations) + self.b
        # Z is a (features, examples) matrix so for every feature,
        # we want the mean and std dev of every row meaning we get
        # a (N,1) vector of means
        self.mean_miniBatch = np.mean(self.Z_in, axis=1, keepdims=True)
        self.variance = np.var(self.Z_in, axis=1, keepdims=True)
        # update running mean and std dev of every feature
        self._update_running_avg()
        # normalize feature means to 0 and std dev to 1
        self.Z_centered = self.Z_in - self.mean_miniBatch
        self.inv_stdDev = 1 / np.sqrt(self.variance + self.eps)
        self.Z_norm = self.Z_centered * self.inv_stdDev
        # Multiply by learnable parameters to avoid the network from
        # losing expressivity
        self.Z_final = self.gamma * self.Z_norm + self.beta
        # Finally, feed into activation function to activate.
        self.A = self.activation_function.compute_output(self.Z_final)
        return self.A

    def _update_running_avg(self, beta: float = 0.9) -> None:
        self.runningMean = (beta) * (self.runningMean) + (
            1 - beta) * self.mean_miniBatch
        self.runningVar = (beta) * (self.runningVar) + (1 -
                                                        beta) * self.variance

    def _test_forward(self, prevlayer_activations: np.ndarray) -> np.ndarray:
        # BatchNorm has diff behaviour at test time and train time
        assert self.W.shape[1] == prevlayer_activations.shape[
            0], "Your weights and inputs shapes are mismatched!"
        z = np.dot(self.W, prevlayer_activations) + self.b
        # Vectorize operation - elementwise subtraction, then
        # elementwise division, where z is a (N,M) matrix and
        # running mean and running variance are (N,1) vectors
        normalized_z = (z - self.runningMean) / (np.sqrt(self.runningVar +
                                                         self.eps))
        # multiply by learnable parameter gamma and add beta
        z_final = self.gamma * normalized_z + self.beta
        return self.activation_function.compute_output(z_final)

    def update_weights(self,
                       dLdA: np.ndarray,
                       learn_rate: float,
                       epoch: int,
                       prediction_obj,
                       curr_x: np.ndarray,
                       curr_y: np.ndarray,
                       layer: int,
                       grad_check: bool = False) -> np.ndarray:
        """ This method computes the backward pass through this layer.
        In the gradient circuit, this layers job is to recieve the
        jacobian matrix of the loss function with respect to this
        layers activations, and update its parameters, and then pass on
        the gradient of the loss function with respect to the previous
        layers activations.

        Args:
            dLdA:
                A numpy array of shape A (nl, M), containing the jacobian matrix
                of the loss function with respect to the activations in this
                layer.

            optim:
                Object repersenting the optimization algorithm to use to
                minimize the objective function

            learn_rate:
                Floating point value representing the learning rate to be used
                when optimizing the objective function

            epoch:
                Integer representing the epoch we are updating for currently

            prediction_obj:
                Object representing the base neural network object we can use

            curr_x:
                Numpy array of examples we are currently training on

            curr_y:
                Numpy array of labels for the examples

            layer:
                Integer representing the layer in the network we are currently
                updating

        Returns:
            A numpy array representing dL/dA_L-1, which is a (nl-1, M)
            numpy array containing the jacobian Matrix of the objective
            function with respect to the activations in the previous layer.
        """
        ## GRADIENT GOING BACKWARDS IN CIRCUIT
        if self.isSoftmax == 0:
            dadz_final = self.activation_function.get_derivative_wrt_input(
                self.Z_final)
            dl_dzfinal = dLdA * dadz_final
        else:
            dl_dzfinal = dl_dz_softmax(self.Z_final, self.A, dLdA,
                                       self.activation_function)

        # gradients for learnable parameters - computation step
        # for z_final layer
        dl_dgamma = np.sum(np.dot(dl_dzfinal, self.Z_norm.T),
                           axis=1,
                           keepdims=True)
        dl_dbeta = np.sum(dl_dzfinal, axis=1, keepdims=True)
        dl_dznorm = dl_dzfinal * self.gamma

        # first branch dl_dz
        dl_dzin_firstbranch = dl_dznorm * self.inv_stdDev
        ## variance portion of dl_dz
        dznorm_dinv = self.Z_centered
        dinv_dvar = -(0.5) * np.power(self.variance + self.eps, -3 / 2)
        dl_dvar = np.sum(dl_dznorm * dznorm_dinv * dinv_dvar,
                         axis=1,
                         keepdims=True)
        dl_dzin_secondbranch = dl_dvar * (2 / self.Z_in.shape[1]) * (
            self.Z_centered)
        ## mean portion of dl_dz
        dl_dmu1 = np.sum(dl_dznorm * (-1) * (self.inv_stdDev),
                         axis=1,
                         keepdims=True)
        dl_dmu2 = np.sum(dl_dvar * (-2 / self.Z_in.shape[1]) *
                         (self.Z_centered),
                         axis=1,
                         keepdims=True)
        dl_dzin_thirdbranch = (1 / self.Z_in.shape[1]) * (dl_dmu1 + dl_dmu2)
        # total dl_dz is sum of all three branches
        dl_dzin = dl_dzin_firstbranch + dl_dzin_secondbranch + dl_dzin_thirdbranch

        # finally get back to the un-normalized activations where
        # we can get what we wanted from the beginning
        dl_dw = np.dot(dl_dzin, self.Ain.T)
        dl_db = np.sum(dl_dzin, axis=1, keepdims=True)
        dl_da_prev_layer = np.dot(self.W.T, dl_dzin)

        assert dl_dw.shape == self.W.shape, (
            "Your W[L] shape is not the same as dW/dW[L] shape")
        assert dl_db.shape == self.b.shape, (
            "Your B[L] shape is not the same as dW/dB[L] shape")
        assert dl_da_prev_layer.shape == self.Ain.shape, (
            "Your dL/dA[L-1] shapes are incomptabile")
        assert dl_dgamma.shape == self.gamma.shape, (
            "Your dL/dGamma shape is not the same as gammas shape")
        assert dl_dbeta.shape == self.beta.shape, (
            "Your dL/dBeta shape is not the same as betas shape")

        # Epoch zero and you want to gradient check, do some gradient checks for
        # params W and b
        if epoch == 0 and grad_check:
            self._gradient_check("W", dl_dw, curr_x, curr_y, prediction_obj,
                                 layer)
            self._gradient_check("b", dl_db, curr_x, curr_y, prediction_obj,
                                 layer)

        ## GRADIENT FROM REGULARIZATION IF REGULARIZATION
        dreg_dw = 0
        if self.regularization != None:
            dreg_dw = self._getRegularizationLoss(self.regularization,
                                                  self.reg_parameter,
                                                  self.Ain.shape[1], self.W)
        dl_dw = dl_dw + dreg_dw
        self.W, self.b, self.gamma, self.beta = self.optim.update_params(
            [self.W, self.b, self.gamma, self.beta],
            [dl_dw, dl_db, dl_dgamma, dl_dbeta],
            learn_rate,
            epoch_num=epoch + 1)
        return dl_da_prev_layer


class DenseDropOutLayer(DenseLayer):

    def __init__(self,
                 num_in,
                 num_layer,
                 activation_function,
                 regularization=None,
                 reg_parameter=None,
                 isSoftmax=0,
                 keepProb=0.5):
        super(DenseDropOutLayer,
              self).__init__(num_in, num_layer, activation_function,
                             regularization, reg_parameter, isSoftmax)
        self.keepProb = keepProb
        # if keep prob is 1, then every value in matrix will be less
        # than 1, and thus be True and so every activated neuron is kept
        # in. If keepProb is 0, then every neuron is dropped.

    def compute_forward(self, x, train=True):
        if train:
            return self._train_forward(x)
        return self._test_forward(x)

    def _train_forward(self, prevlayer_activations):
        assert self.W.shape[1] == prevlayer_activations.shape[
            0], "Your weights and inputs shapes are mismatched!"
        self.Ain = prevlayer_activations
        self.Z = np.dot(self.W, prevlayer_activations) + self.b
        self.mask = (np.random.rand(*self.Z.shape) < self.keepProb).astype(int)
        A_undropped = self.activation_function.compute_output(self.Z)
        A_dropped = A_undropped * self.mask
        # have to divide by keepProb because we don't want the
        # activations of the next layer to have lower values just
        # cause we're feeding in so many values that are zero
        self.A = A_dropped / self.keepProb
        return self.A

    def _test_forward(self, prevlayer_activations):
        # testing forward has no dropout - we use our full architecture
        # so its just the same as forward prop through a dense layer
        assert self.W.shape[1] == prevlayer_activations.shape[
            0], "Your weights and inputs shapes are mismatched!"
        Ain = prevlayer_activations
        z = np.dot(self.W, prevlayer_activations) + self.b
        a = self.activation_function.compute_output(self.Z)
        return a

    def update_weights(self,
                       dLdA,
                       learn_rate,
                       epoch,
                       prediction_obj,
                       curr_x,
                       curr_y,
                       layer,
                       grad_check=False):
        """
        This method computes the backward pass through this layer. In the gradient circuit,
        this layers job is to recieve the jacobian matrix of the loss function with respect
        to this layers activations, and update its parameters, and then pass on the gradient of the 
        loss function with respect to the previous layers activations. 


        Parameters:
        -> dLdA (NumPy matrix) -> A (nl, M) NumPy matrix containing the 
        jacobian matrix of the loss function with respect to the activations 
        in this layer.

        -> optim (function) -> Optimizer to use to minimize the loss function 

        -> learn_rate (float) -> learning rate to be used when optimizing the cost function

        -> epoch (int) -> the epoch we are updating for currently
        
        -> prediction_obj (NeuralNet obj) -> the base neural network object we can use

        -> curr_x (NumPy matrix) -> matrix of examples we are currently training on

        -> curr_y (NumPy vector) -> matrix of labels for the examples

        -> layer (int) -> layer in the network we are currently updating 

        Returns: dL/dA_L-1 (NumPy Matrix) -> A (nl-1, M) NumPy matrix containing the jacobian Matrix
        of the loss function with respect to the activations in the previous layer.
        """

        ## GRADIENT GOING BACKWARDS IN CIRCUIT
        # you dont use a dropout layer w/ a softmax activation

        # only change between dropout and normal dense layer - you have the
        # boolean mask and /p term when getting dadz
        dadz = (self.activation_function.get_derivative_wrt_input(self.Z) *
                self.mask) / (self.keepProb)
        dl_dz = dLdA * dadz

        dl_dw = np.dot(dl_dz, self.Ain.T)
        dl_db = np.sum(dl_dz, axis=1, keepdims=True)
        dl_da_prev_layer = np.dot(self.W.T, dl_dz)

        assert dl_dw.shape == self.W.shape, (
            "Your W[L] shape is not the same as dW/dW[L] shape")
        assert dl_db.shape == self.b.shape, (
            "Your B[L] shape is not the same as dW/dB[L] shape")
        assert dl_da_prev_layer.shape == self.Ain.shape, (
            "Your dL/dA[L-1] shapes are incomptabile")

        # Epoch zero and you want to gradient check, do some gradient checks
        # for params W and b
        if epoch == 0 and grad_check:
            self._gradient_check("W", dl_dw, curr_x, curr_y, prediction_obj,
                                 layer)
            self._gradient_check("b", dl_db, curr_x, curr_y, prediction_obj,
                                 layer)

        ## GRADIENT FROM REGULARIZATION IF REGULARIZATION
        dreg_dw = 0
        if self.regularization != None:
            dreg_dw = self._getRegularizationLoss(self.regularization,
                                                  self.reg_parameter,
                                                  self.Ain.shape[1], self.W)
        dl_dw = dl_dw + dreg_dw
        self.W, self.b = self.optim.update_params([self.W, self.b],
                                                  [dl_dw, dl_db],
                                                  learn_rate,
                                                  epoch_num=epoch + 1)
        return dl_da_prev_layer
