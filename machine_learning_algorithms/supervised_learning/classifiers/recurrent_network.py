""" This module contains code representing a reccurrent neural network used for
the natural language processing task of language modelling."""
import numpy as np
from machine_learning_algorithms.utility.misc import one_hot_encode_feature
from machine_learning_algorithms.neural_net_utility.reccurent_neural_net_layers import ReccurentNetCellGeneration
from machine_learning_algorithms.neural_net_utility.optimizer import AdaGrad, Optimizer
from machine_learning_algorithms.neural_net_utility.activation_functions import BaseActivationFunction
from typing import Dict, Tuple, List, Union


class ReccurentNetLanguageModel:
    """ This class represents a Recurrent Neural Network(RNN) with a
    many-to-many architecture used for the natural language processing
    task of language modelling.

    Attributes:
        idx_to_char:
            Dictionary containing a mapping between integer keys to characters

        char_to_idx:
            Dictionary containing a mapping between strings to integers

        activation_function:
            Object of type BaseActivationFunction representing the activation
            function to be used in the net

        number_neurons:
            Integer representing the total number of neurons in the RNN

        number_features:
            Integer representing the total number of features being input to the
            network

        temperature:
            Floating point value indicating how much randomness to inject
            into the networks predictions

    """

    def __init__(self,
                 idx_to_char: Dict[int, str],
                 char_to_idx: Dict[str, int],
                 activation_function: BaseActivationFunction,
                 number_neurons: int,
                 number_features: int,
                 temperature: int = 1):
        self.number_features = number_features
        self.number_neurons = number_neurons
        self.model = ReccurentNetCellGeneration(
            num_neurons=number_neurons,
            activation_function=activation_function,
            num_input_features=number_features)
        self.idx_to_char = idx_to_char
        self.char_to_idx = char_to_idx
        self.temperature = temperature

    def fit(
        self,
        xtrain: str,
        time_steps_unroll: int,
        xvalid: str = None,
        num_epochs: int = 10,
        ret_train_loss: bool = False,
        learn_rate: float = 0.01,
        optim: Optimizer = AdaGrad(),
        verbose: bool = False
    ) -> Union[None, Tuple[List[float], List[float]], List[float]]:
        """ This method trains the reccurrent net language model on the given
        dataset.

        Args:
            xtrain:
                String representing a text file containing the data the network
                should emulate

            xvalid:
                String representing a text file containing the data the network
                will be validated on

            time_steps_unroll:
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
                is AdaGrad

            verbose:
                Boolean value indicating whether to provide updates when
                training. If True, will get indications when each epoch is
                is done training along with a sample generated sentence

        Returns:
            None if ret_train_loss is False. If ret_train_loss is True and a
            numpy array is passed as input for x_valid, a tuple will
            be returned consisting of two lists of floating point values,
            one representing the training losses, the other representing
            the validation losses. If just ret_train_loss is True, then
            a list of floating point values representing the training losses
            will be returned.
        """
        train_loss = []
        valid_loss = []
        for epoch in range(num_epochs):
            curr_sample_start_train = 0
            # cycle through the entire training set
            loss_epoch = []
            while curr_sample_start_train + time_steps_unroll < len(xtrain):
                # the label for a given char is the char right after it
                slice_x = xtrain[
                    curr_sample_start_train:curr_sample_start_train +
                    time_steps_unroll]
                slice_y = xtrain[curr_sample_start_train +
                                 1:curr_sample_start_train + time_steps_unroll +
                                 1]
                # new batch
                curr_sample_start_train += time_steps_unroll
                activations_prev = np.zeros((self.number_neurons, 1))
                # forward pass
                loss_epoch.append(
                    self.model.train_forward(slice_x, slice_y, activations_prev,
                                             self.char_to_idx))
                # backward pass
                self.model.update_weights(learn_rate, time_steps_unroll,
                                          slice_y, self.char_to_idx, epoch,
                                          optim)
                if verbose:
                    print(loss_epoch[-1])

            train_loss.append(np.mean(loss_epoch))
            if xvalid:
                valid_loss.append(
                    self._get_valid_loss(xvalid, time_steps_unroll))
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
        seed_idx = np.random.randint(0, high=self.number_features)
        seed_vector = one_hot_encode_feature(self.number_features, seed_idx)
        a_prev = np.zeros((self.number_neurons, 1))
        return self.model.generate(seed_vector, a_prev, total_generating_steps,
                                   self.idx_to_char, self.temperature)

    def _get_valid_loss(self, xvalid: str, time_steps_unroll: int) -> float:
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
            activations_prev = np.zeros((self.number_neurons, 1))
            loss.append(
                self.model.train_forward(slice_x,
                                         slice_y,
                                         activations_prev,
                                         self.char_to_idx,
                                         cache=False))
        return np.mean(loss)
