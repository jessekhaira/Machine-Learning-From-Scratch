""" This module contains code representing a reccurrent neural network used for
the natural language processing task of language modelling."""
import numpy as np
from machine_learning_algorithms.Utility.misc import oneHotEncodeFeature
from machine_learning_algorithms.neural_net_utility.reccurent_neural_net_layers import RNN_cell_languageModel
from machine_learning_algorithms.neural_net_utility.optimizer import AdaGrad, Optimizer
from machine_learning_algorithms.neural_net_utility.activation_functions import BaseActivationFunction
from typing import Dict


class ReccurentNetLanguageModel:
    """ This class represents a Recurrent Neural Network(RNN) with a
    many-to-many architecture used for the natural language processing
    task of language modelling.

    Attributes:
        idx_to_char:
            Dictionary containing a mapping between integer keys to characters

        char_to_idx:
            Dictionary containing a mapping between strings to integers

        activationFunction:
            Object of type BaseActivationFunction representing the activation
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
                 idx_to_char: Dict[int, str],
                 char_to_idx: Dict[str, int],
                 activationFunction: BaseActivationFunction,
                 numberNeurons: int,
                 numberFeatures: int,
                 temperature: int = 1):
        self.numberFeatures = numberFeatures
        self.numberNeurons = numberNeurons
        self.model = RNN_cell_languageModel(
            numNeurons=numberNeurons,
            activationFunctionLayer=activationFunction,
            numInputFeatures=numberFeatures)
        self.idx_to_char = idx_to_char
        self.char_to_idx = char_to_idx
        self.temperature = temperature

    def fit(self,
            xtrain: str,
            timeStepsUnroll: int,
            xvalid: str = None,
            num_epochs: int = 10,
            ret_train_loss: bool = False,
            learn_rate: float = 0.01,
            optim: Optimizer = AdaGrad(),
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
                Object of type Optimizer representing the optimization algorithm
                to use to minimize the loss function. Default Optimizer to use
                is AdaGrad.

            verbose:
                Boolean value indicating whether to provide updates when
                training. If True, will get indications when each epoch is
                is done training along with a sample generated sentence.
        """
        train_loss = []
        valid_loss = []
        for epoch in range(num_epochs):
            curr_sample_start_train = 0
            # cycle through the entire training set
            loss_epoch = []
            while curr_sample_start_train + timeStepsUnroll < len(xtrain):
                # the label for a given char is the char right after it
                slice_x = xtrain[
                    curr_sample_start_train:curr_sample_start_train +
                    timeStepsUnroll]
                slice_y = xtrain[curr_sample_start_train +
                                 1:curr_sample_start_train + timeStepsUnroll +
                                 1]
                # new batch
                curr_sample_start_train += timeStepsUnroll
                activations_prev = np.zeros((self.numberNeurons, 1))
                # forward pass
                loss_epoch.append(
                    self.model._train_forward(slice_x, slice_y,
                                              activations_prev,
                                              self.char_to_idx))
                # backward pass
                self.model.update_weights(learn_rate, timeStepsUnroll, slice_y,
                                          self.char_to_idx, epoch, optim)
                if verbose:
                    print(loss_epoch[-1])

            train_loss.append(np.mean(loss_epoch))
            if xvalid:
                valid_loss.append(self._get_valid_loss(xvalid, timeStepsUnroll))
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

    def generate(self, total_generating_steps: int = 200) -> str:
        """ This method generates text from the RNN.

        Args:
            total_generating_steps:
                Integer representing the length of the sequence that
                should be generated

        Returns:
            A string of length total_generating_steps that represents
            text generated from the RNN.
        """
        seed_idx = np.random.randint(0, high=self.numberFeatures)
        seed_vector = oneHotEncodeFeature(self.numberFeatures, seed_idx)
        a_prev = np.zeros((self.numberNeurons, 1))
        return self.model.generate(seed_vector, a_prev, total_generating_steps,
                                   self.idx_to_char, self.temperature)

    def _get_valid_loss(self, xvalid: str, time_steps_unroll: int):
        loss = []
        curr_sample_start_valid = 0
        while curr_sample_start_valid + time_steps_unroll < len(xvalid):
            # the label for a given char is the char right after it
            slice_x = xvalid[curr_sample_start_valid:curr_sample_start_valid +
                             time_steps_unroll]
            slice_y = xvalid[curr_sample_start_valid +
                             1:curr_sample_start_valid + time_steps_unroll + 1]
            # new batch
            curr_sample_start_valid += time_steps_unroll
            activations_prev = np.zeros((self.numberNeurons, 1))
            loss.append(
                self.model._train_forward(slice_x,
                                          slice_y,
                                          activations_prev,
                                          self.char_to_idx,
                                          cache=False))
        return np.mean(loss)
