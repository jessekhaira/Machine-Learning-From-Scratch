import sys
sys.path.append("/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Neural_Net_Util")
sys.path.append("/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Utility")
import numpy as np
from RecurrentNetLayers import RNN_cell_languageModel
from NeuralNetwork_Base import NeuralNetwork_Base
from ActivationFunctions import TanH
from LossFunctions import cross_entropy
from ConvolutionalLayers import Conv2D
from ConvolutionalLayers import Pool
from Optimizers import gradientDescent
from Optimizers import AdaGrad
from misc import oneHotEncodeFeature
import copy 

class ReccurentNet_languageModelChar(object):
    """
    This class represents Recurrent language modelling at the character level Networks.

    Parameters:
    -> idxToChar (HashTable<Integer, Char>): HashTable mapping integer keys to characters

    -> charToIdx (HashTable<Char, Integer>): HashTable mapping chars to integers

    -> activationFunction (obj): Object representing the activation function used in the net

    -> numberNeurons (int): Integer representing the total number of neurons in the RNN

    -> numberFeatures (int): The total number of features being input to the network 

    """
    def __init__(self, idxToChar, charToIdx, activationFunction, numberNeurons, numberFeatures, temperature = 1):
        self.numberFeatures = numberFeatures
        self.numberNeurons = numberNeurons
        self.model = RNN_cell_languageModel(numNeurons = numberNeurons, activationFunctionLayer = activationFunction, numInputFeatures = numberFeatures)
        self.idxToChar  = idxToChar
        self.charToIdx = charToIdx 
        self.temperature = temperature

    def fit(self, xtrain, timeStepsUnroll, xvalid = None, num_epochs=10, ret_train_loss = False, learn_rate = 0.01, optim = AdaGrad(), verbose = False):
        """
        This method trains the reccurrent net language model on the given dataset. 

        Parameters:
        -> xtrain (txt file): Text file containing the data the RNN should emulate

        -> xvalid (txt file): Text file containing the data the RNN should emulate

        -> timeStepsUnroll(int): The length of a sequence from the text file that will be used 

        -> num_epochs (int): Number of epochs to train the model

        -> ret_train_loss (Boolean): Boolean value indicating whether an array of the expected values
        of the loss per epoch should be returned 

        -> learn_rate (float): learning rate to be used when optimizing the loss function

        -> optim (function): optimizer to use to minimize the loss function 

        -> verbose (boolean): boolean value indicating whether to provide updates when training. If True, 
        will get indications when each epoch is done training along with a sample generated sentence. 

        Returns: None
        """
        train_loss = []
        valid_loss = [] 
        print(len(xtrain))
        for epoch in range(num_epochs):
            currSampleStartTrain = 0
            # cycle through the entire training set 
            loss_epoch = [] 
            while currSampleStartTrain + timeStepsUnroll < len(xtrain):
                # the label for a given char is the char right after it
                slice_x = xtrain[currSampleStartTrain:currSampleStartTrain+timeStepsUnroll]
                slice_y = xtrain[currSampleStartTrain+1: currSampleStartTrain+timeStepsUnroll+1]
                # new batch
                currSampleStartTrain += timeStepsUnroll
                activations_prev = np.zeros((self.numberNeurons,1))
                # forward pass 
                loss_epoch.append(self.model._train_forward(slice_x, slice_y, activations_prev, self.charToIdx))
                # backward pass 
                self.model._updateWeights(learn_rate, timeStepsUnroll, slice_y, self.charToIdx, epoch, optim)
                if verbose:
                    print(loss_epoch[-1])

            train_loss.append(np.mean(loss_epoch))
            if xvalid:
                valid_loss.append(self._getValidLoss(xvalid, timeStepsUnroll))
            if verbose and epoch%10 == 0:
                print("Finish epoch %s, train loss: %s"%(epoch, train_loss[-1]))
                if xvalid:
                    print("valid loss: %s"%(valid_loss[-1]))

                print("Generating sentences:")
                print(self.generate())

        if ret_train_loss and xvalid:
            return train_loss, valid_loss
        elif ret_train_loss:
            return train_loss
            

    def generate(self, totalGeneratingSteps = 200):
        """
        This method generates text from the RNN.

        Parameters:
        -> totalGeneratingSteps (int): Integer representing the length of the sequence that should be generated
        """
        seedIdx = np.random.randint(0, high = self.numberFeatures)
        seedVector = oneHotEncodeFeature(self.numberFeatures, seedIdx)
        a_prev = np.zeros((self.numberNeurons, 1))
        return self.model.generate(seedVector, a_prev, totalGeneratingSteps, self.idxToChar, self.temperature)


    def _getValidLoss(self, xvalid, timeStepsUnroll):
        loss = [] 
        currSampleStartValid = 0 
        while currSampleStartValid + timeStepsUnroll < len(xvalid):
            # the label for a given char is the char right after it
            slice_x = xvalid[currSampleStartValid:currSampleStartValid+timeStepsUnroll]
            slice_y = xvalid[currSampleStartValid+1: currSampleStartValid+timeStepsUnroll+1]
            # new batch
            currSampleStartValid += timeStepsUnroll
            activations_prev = np.zeros((self.numberNeurons,1))
            loss.append(self.model._train_forward(slice_x, slice_y, activations_prev, self.charToIdx, cache=False))
        return np.mean(loss) 




            



