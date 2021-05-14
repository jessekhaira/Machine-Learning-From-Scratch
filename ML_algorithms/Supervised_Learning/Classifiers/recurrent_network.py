""" This module contains code representing a reccurrent neural network used for
the natural language processing task of language modelling."""
import numpy as np
from ML_algorithms.Neural_Net_Util.RecurrentNetLayers import RNN_cell_languageModel
from ML_algorithms.Neural_Net_Util.Optimizers import AdaGrad, optimizer
from ML_algorithms.Neural_Net_Util.ActivationFunctions import Base_ActivationFunction
from typing import Dict


class ReccurentNetLanguageModel:
    """ This class represents a Recurrent Neural Network(RNN) with a
    many-to-many architecture used for the natural language processing
    task of language modelling.

    Attributes:
        idxToChar:
            Dictionary containing a mapping between integer keys to characters

        charToIdx:
            Dictionary containing a mapping between strings to integers

        activationFunction:
            Object of type Base_ActivationFunction representing the activation
            function to be used in the net

        numberNeurons:
            Integer representing the total number of neurons in the RNN

        numberFeatures:
            Integer representing the total number of features being input to the
            network

        temperature:
            Floating point value indicating how much randomness to inject
            into the networks predictions

    """

    def __init__(self,
                 idxToChar: Dict[int, str],
                 charToIdx: Dict[str, int],
                 activationFunction: Base_ActivationFunction,
                 numberNeurons: int,
                 numberFeatures: int,
                 temperature: int = 1):
        self.numberFeatures = numberFeatures
        self.numberNeurons = numberNeurons
        self.model = RNN_cell_languageModel(
            numNeurons=numberNeurons,
            activationFunctionLayer=activationFunction,
            numInputFeatures=numberFeatures)
        self.idxToChar = idxToChar
        self.charToIdx = charToIdx
        self.temperature = temperature

    def fit(self,
            xtrain: str,
            timeStepsUnroll: int,
            xvalid: str = None,
            num_epochs: int = 10,
            ret_train_loss: bool = False,
            learn_rate: float = 0.01,
            optim: optimizer = AdaGrad(),
            verbose: bool = False) -> None:
        """ This method trains the reccurrent net language model on the given
        dataset.

        Args:
            xtrain:
                String representing a text file containing the data the network
                should emulate

            xvalid:
                String representing a text file containing the data the network
                will be validated on

            timeStepsUnroll:
                Integer representing the length of a sequence from the text
                file that will be used during training

            num_epochs:
                Integer representing the number of epochs to train the model

            ret_train_loss:
                Boolean value indicating whether an array of the expected values
                of the loss per epoch should be returned

            learn_rate:
                Floating point value representing the learning rate to be
                used when optimizing the loss function

            optim:
                Object of type optimizer representing the optimization algorithm
                to use to minimize the loss function. Default optimizer to use
                is AdaGrad.

            verbose:
                Boolean value indicating whether to provide updates when
                training. If True, will get indications when each epoch is
                is done training along with a sample generated sentence.
        """
        train_loss = []
        valid_loss = []
        for epoch in range(num_epochs):
            currSampleStartTrain = 0
            # cycle through the entire training set
            loss_epoch = []
            while currSampleStartTrain + timeStepsUnroll < len(xtrain):
                # the label for a given char is the char right after it
                slice_x = xtrain[currSampleStartTrain:currSampleStartTrain +
                                 timeStepsUnroll]
                slice_y = xtrain[currSampleStartTrain + 1:currSampleStartTrain +
                                 timeStepsUnroll + 1]
                # new batch
                currSampleStartTrain += timeStepsUnroll
                activations_prev = np.zeros((self.numberNeurons, 1))
                # forward pass
                loss_epoch.append(
                    self.model._train_forward(slice_x, slice_y,
                                              activations_prev, self.charToIdx))
                # backward pass
                self.model._updateWeights(learn_rate, timeStepsUnroll, slice_y,
                                          self.charToIdx, epoch, optim)
                if verbose:
                    print(loss_epoch[-1])

            train_loss.append(np.mean(loss_epoch))
            if xvalid:
                valid_loss.append(self._getValidLoss(xvalid, timeStepsUnroll))
            if verbose and epoch % 10 == 0:
                print("Finish epoch %s, train loss: %s" %
                      (epoch, train_loss[-1]))
                if xvalid:
                    print("valid loss: %s" % (valid_loss[-1]))

                print("Generating sentences:")
                print(self.generate())

        if ret_train_loss and xvalid:
            return train_loss, valid_loss
        elif ret_train_loss:
            return train_loss

    def generate(self, totalGeneratingSteps=200):
        """
        This method generates text from the RNN.

        Parameters:
        -> totalGeneratingSteps (int): Integer representing the length of the sequence that should be generated
        """
        seedIdx = np.random.randint(0, high=self.numberFeatures)
        seedVector = oneHotEncodeFeature(self.numberFeatures, seedIdx)
        a_prev = np.zeros((self.numberNeurons, 1))
        return self.model.generate(seedVector, a_prev, totalGeneratingSteps,
                                   self.idxToChar, self.temperature)

    def _getValidLoss(self, xvalid, timeStepsUnroll):
        loss = []
        currSampleStartValid = 0
        while currSampleStartValid + timeStepsUnroll < len(xvalid):
            # the label for a given char is the char right after it
            slice_x = xvalid[currSampleStartValid:currSampleStartValid +
                             timeStepsUnroll]
            slice_y = xvalid[currSampleStartValid + 1:currSampleStartValid +
                             timeStepsUnroll + 1]
            # new batch
            currSampleStartValid += timeStepsUnroll
            activations_prev = np.zeros((self.numberNeurons, 1))
            loss.append(
                self.model._train_forward(slice_x,
                                          slice_y,
                                          activations_prev,
                                          self.charToIdx,
                                          cache=False))
        return np.mean(loss)
