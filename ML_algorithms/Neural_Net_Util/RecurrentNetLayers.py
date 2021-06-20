import numpy as np
import random
from ML_algorithms.Neural_Net_Util.neural_net_layers import _BaseLayer
import math
from ML_algorithms.Utility.misc import gradientClipping
from ML_algorithms.Utility.misc import oneHotEncodeFeature
from ML_algorithms.Utility.misc import convertToHighestPred
from ML_algorithms.Neural_Net_Util.ActivationFunctions import Softmax
from ML_algorithms.Neural_Net_Util.LossFunctions import cross_entropy
from ML_algorithms.Neural_Net_Util.Optimizers import AdaGrad


class RNN_cell_languageModel(_BaseLayer):
    """
    This class represents a single RNN cell performing language generation. This RNN forms a 
    many-to-many architecture, as the cell will predict a letter at every single time step.

    Parameters:
    -> numNeurons (int): The number of neurons present in this cell
    -> numInputFeatures (int): The number of features input to the cell at every time step
    -> activationFunctionLayer (obj): The activation function to be used to introduce non-linearity
    into the network
    """

    def __init__(self, numNeurons, numInputFeatures, activationFunctionLayer):
        self.numNeurons = numNeurons
        self.numInputFeatures = numInputFeatures
        self.Waa = self._initalizeWeights(numNeurons, numNeurons)
        self.Wax = self._initalizeWeights(numNeurons, numInputFeatures)
        self.ba = self._initalizeBias(numNeurons)
        self.Way = self._initalizeWeights(numInputFeatures, numNeurons)
        self.by = self._initalizeBias(numInputFeatures)
        self.activationFunctionLayer = activationFunctionLayer

        # we're only going to have one RNN cell to do all the generating IE if
        # we have a 100 word long sentence, we don't make a new RNN cell for every single word
        # so we need to cache all the activations and predictions at every time step
        # so we can backprop
        self.cache_perTimestep = {}

    def _train_forward(self, x, y, a_prev, charToIdx, cache=True):
        """
        This method computes the forward step for a recurrent neural network layer
        performing language modelling. The RNN will be unrolled for t time steps
        and compute a loss over those t time steps.

        Parameters:
        -> x (txt file): The text that the RNN should emulate
        -> y (txt file): The labels that the RNN should predict
        -> a_prev (NumPy vector): Vector containing the activations from the previous
        time step
        -> charToIdx (HashTable): HashTable mapping characters to indices 

        Returns:
        -> loss (int): The loss accumulated over the t timesteps
        """
        # loop through every time step
        loss = 0
        for t in range(len(x)):
            char = x[t]
            label = y[t]
            indexTrain = charToIdx[char]
            indexLabel = charToIdx[label]
            # now generate one hot vector w/ this char
            x_t = oneHotEncodeFeature(self.numInputFeatures, indexTrain)
            y_t = oneHotEncodeFeature(self.numInputFeatures, indexLabel)
            # feed the vector into the net
            A, predictions, a_prev, x_t, Z_activ = self._compute_OneStepForward(
                x_t, a_prev)
            # pass on the activations from this layer to the next
            a_prev = A
            # see how well we did with the prediction, accumulate loss for every time step
            loss += cross_entropy().get_loss(y_t, predictions)
            if cache:
                self.cache_perTimestep[t] = (A, predictions, a_prev, x_t,
                                             Z_activ)

        return loss

    def _compute_OneStepForward(self, x_t, a_prev, temperature=1):
        assert self.Wax.shape[1] == x_t.shape[0]
        assert a_prev.shape[0] == self.Waa.shape[1]
        Z_activ = np.dot(self.Wax, x_t) + np.dot(self.Waa, a_prev) + self.ba
        A = self.activationFunctionLayer.compute_output(Z_activ)

        # raw logits/temp = input to activation function
        Z_pred = (np.dot(self.Way, A) + self.by) / temperature

        assert Z_pred.shape == (self.numInputFeatures, 1)
        predictions = Softmax().compute_output(Z_pred)
        return A, predictions, a_prev, x_t, Z_activ

    def _updateWeights(self, learn_rate, total_timeSteps, y_label, charToIdx,
                       epochNum, optim):
        """
        This method carries out backpropagation through time for an RNN cell performing
        language modelling.

        Parameters:
        -> learn_rate (int): Integer representing how much to update the weights in gradient descent
        -> total_timeSteps (int): The total number of timesteps this RNN will be unrolled for.
        -> y_label (NumPy vector): (N,1) NumPy vector with labels indicating the correct prediction at each time
        step t in the RNN cell

        Returns:
        -> None 
        """
        # this is different from normal neural nets and conv nets
        # where every layer was a seperate entity and each layers responsibility
        # was to recieve dLdA and update its weights (if there were any) and then
        # pass back dL/dA_prev. The key thing with RNN's is there is a horizontal temporal dimension
        # - so one RNN cell is unrolled for total_timeSteps. We can go deeper as well though
        # and pass the output from this RNN cell into another RNN cell another layer deep.

        # init everything
        dLdA_layerAhead = np.zeros((self.numNeurons, 1))
        dLdBy, dLdWay = np.zeros_like(self.by), np.zeros_like(self.Way)
        dLdBa, dLdWaa, dLdWax = np.zeros_like(self.ba), np.zeros_like(
            self.Waa), np.zeros_like(self.Wax)

        # backprop through time :D
        for t in reversed(range(total_timeSteps)):
            A = self.cache_perTimestep[t][0]
            predictions = self.cache_perTimestep[t][1]
            Z_activ = self.cache_perTimestep[t][4]

            A_prev = self.cache_perTimestep[t][2]
            x_t = self.cache_perTimestep[t][3]
            # combine dL/dy^ and dy^/dz to get dL/dZ for cross entropy loss and
            # softmax activation - very quick and efficient
            label = y_label[t]
            int_label = charToIdx[label]
            dLdZ_pred = predictions
            dLdZ_pred[int_label] -= 1

            # The same weights are used for every single time step predictions
            # therefore the gradients must accumulate for all the parameters!

            # updates for the softmax classifier
            dLdBy += dLdZ_pred
            dLdWay += np.dot(dLdZ_pred, A.T)

            # this activation goes to the cell in the next time step and is used again
            # so it gets a gradient portion directly from the softmax classifier and from the next time step
            dLdA = np.dot(self.Way.T, dLdZ_pred) + dLdA_layerAhead
            # backprop through the nonlinearity to get dLdZ_activ
            dadZ_activ = self.activationFunctionLayer.getDerivative_wrtInput(
                Z_activ)
            dLdZ_activ = dLdA * dadZ_activ

            dLdBa += dLdZ_activ
            dLdWaa += np.dot(dLdZ_activ, A_prev.T)
            dLdWax += np.dot(dLdZ_activ, x_t.T)

            # pass gradient back to prev activation
            dLdA_layerAhead = np.dot(self.Waa, dLdA)

        # RNN's can suffer from exploding gradients, so before we update the params
        # we are going to do gradient clipping
        params = [self.Way, self.Waa, self.Wax, self.ba, self.by]
        dparams = [dLdWay, dLdWaa, dLdWax, dLdBa, dLdBy]
        gradientClipping(dparams)
        self.Way, self.Waa, self.Wax, self.ba, self.by = optim.updateParams(
            params, dparams, learn_rate, epochNum)

    def generate(self, seedVector, a_prev, totalGeneratingSteps, idxToChar,
                 temperature):
        """
        This method generates sequences from the RNN. 

        Parameters:
        -> seedVector (NumPy vector): Vector containing a seed character for the RNN. Can be randomly chosen.
        -> a_prev (NumPy vector): Vector of previous activations, should be zero to start with. 
        -> totalGeneratingSteps (int): The length of the sequence that the user wants to generate
        -> idxToChar (HashTable<Integer,String>): A mapping between integer indices to characters

        Returns:
        -> output (string): A string indicating the words the RNN predicted 
        """
        x_t = seedVector
        output = []
        for i in range(totalGeneratingSteps):
            A, predicted_vector, _, _, _ = self._compute_OneStepForward(
                x_t, a_prev, temperature)
            # lets choose a letter weighted by the probability given by the predicted vector
            predicted_index = np.random.choice(self.numInputFeatures,
                                               p=predicted_vector.ravel())
            # the predictiion from this timestep is the input for the next time step
            x_t = predicted_vector
            a_prev = A
            # get the char that the index represents and join it to the output
            predictedLetter = idxToChar[predicted_index]
            output.append(predictedLetter)
        return "".join(output)

    def _initalizeWeights(self, size1, size2):
        # sample from a standard normal distribution and multiple by 0.01
        # to init weights
        return np.random.randn(size1, size2) * 0.01

    def _initalizeBias(self, size1):
        return np.zeros((size1, 1))
