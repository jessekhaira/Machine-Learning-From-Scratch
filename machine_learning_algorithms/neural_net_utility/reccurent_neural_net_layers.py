import numpy as np
from machine_learning_algorithms.neural_net_utility.neural_net_layers import BaseNeuralNetworkLayer
from machine_learning_algorithms.utility.misc import gradientClipping
from machine_learning_algorithms.utility.misc import oneHotEncodeFeature
from machine_learning_algorithms.neural_net_utility.activation_functions import Softmax, BaseActivationFunction
from machine_learning_algorithms.neural_net_utility.loss_functions import cross_entropy


class ReccurentNetCellGeneration(BaseNeuralNetworkLayer):
    """ This class represents a single RNN cell performing language
    generation. This RNN forms a many-to-many architecture, as the cell
    will predict a letter at every single time step.

    Attributes:
        num_neurons:
            Integer representing the number of neurons present in this cell

        num_input_features:
            Integer representing the number of features input to the cell at
            every time step

        activation_function:
            Object of type BaseActivationFunction that represents the activation
            function to be used to introduce non-linearity into the network
    """

    def __init__(self, num_neurons: int, num_input_features: int,
                 activation_function: BaseActivationFunction):
        self.num_neurons = num_neurons
        self.num_input_features = num_input_features
        self.waa = self._initialize_weights(num_neurons, num_neurons)
        self.wax = self._initialize_weights(num_neurons, num_input_features)
        self.ba = self._initialize_bias(num_neurons)
        self.way = self._initialize_weights(num_input_features, num_neurons)
        self.by = self._initialize_bias(num_input_features)
        self.activation_function = activation_function

        # we're only going to have one RNN cell to do all the generating IE if
        # we have a 100 word long sentence, we don't make a new RNN cell for
        # every single word so we need to cache all the activations and
        # predictions at every time step so we can backprop
        self.cache_per_time_step = {}

    def train_forward(self, x, y, a_prev, char_to_idx, cache=True):
        """
        This method computes the forward step for a recurrent neural network layer
        performing language modelling. The RNN will be unrolled for t time steps
        and compute a loss over those t time steps.

        Parameters:
        -> x (txt file): The text that the RNN should emulate
        -> y (txt file): The labels that the RNN should predict
        -> a_prev (NumPy vector): Vector containing the activations from the previous
        time step
        -> char_to_idx (HashTable): HashTable mapping characters to indices

        Returns:
        -> loss (int): The loss accumulated over the t timesteps
        """
        # loop through every time step
        loss = 0
        for t in range(len(x)):
            char = x[t]
            label = y[t]
            index_train = char_to_idx[char]
            index_label = char_to_idx[label]
            # now generate one hot vector w/ this char
            x_t = oneHotEncodeFeature(self.num_input_features, index_train)
            y_t = oneHotEncodeFeature(self.num_input_features, index_label)
            # feed the vector into the net
            a, predictions, a_prev, x_t, z_activ = self._compute_one_step_forward(
                x_t, a_prev)
            # pass on the activations from this layer to the next
            a_prev = a
            # see how well we did with the prediction, accumulate loss for
            # every time step
            loss += cross_entropy().get_loss(y_t, predictions)
            if cache:
                self.cache_per_time_step[t] = (a, predictions, a_prev, x_t,
                                               z_activ)

        return loss

    def _compute_one_step_forward(self, x_t, a_prev, temperature=1):
        assert self.wax.shape[1] == x_t.shape[0]
        assert a_prev.shape[0] == self.waa.shape[1]
        z_activ = np.dot(self.wax, x_t) + np.dot(self.waa, a_prev) + self.ba
        a = self.activation_function.compute_output(z_activ)

        # raw logits/temp = input to activation function
        z_pred = (np.dot(self.way, a) + self.by) / temperature

        assert z_pred.shape == (self.num_input_features, 1)
        predictions = Softmax().compute_output(z_pred)
        return a, predictions, a_prev, x_t, z_activ

    def update_weights(self, learn_rate, total_timeSteps, y_label, char_to_idx,
                       epochNum, optim):
        """
        This method carries out backpropagation through time for an RNN cell performing
        language modelling.

        Parameters:
        -> learn_rate (int): Integer representing how much to update the weights in gradient descent
        -> total_timeSteps (int): The total number of timesteps this RNN will be unrolled for.
        -> y_label (NumPy vector): (N,1) NumPy vector with labels indicating the correct prediction at each time
        step t in the RNN cell
        """
        # this is different from normal neural nets and conv nets
        # where every layer was a seperate entity and each layers responsibility
        # was to recieve dl_da and update its weights (if there were any) and
        # then pass back dL/dA_prev. The key thing with RNN's is there is a
        # horizontal temporal dimension - so one RNN cell is unrolled for
        # total_timeSteps. We can go deeper as well though and pass the output
        # from this RNN cell into another RNN cell another layer deep.

        # init everything
        dl_da_layer_ahead = np.zeros((self.num_neurons, 1))
        dl_dby, dl_dway = np.zeros_like(self.by), np.zeros_like(self.way)
        dl_dba, dl_dwaa, dl_dwax = np.zeros_like(self.ba), np.zeros_like(
            self.waa), np.zeros_like(self.wax)

        # backprop through time :D
        for t in reversed(range(total_timeSteps)):
            a = self.cache_per_time_step[t][0]
            predictions = self.cache_per_time_step[t][1]
            z_activ = self.cache_per_time_step[t][4]

            a_prev = self.cache_per_time_step[t][2]
            x_t = self.cache_per_time_step[t][3]
            # combine dL/dy^ and dy^/dz to get dL/dZ for cross entropy loss and
            # softmax activation - very quick and efficient
            label = y_label[t]
            int_label = char_to_idx[label]
            dl_dzpred = predictions
            dl_dzpred[int_label] -= 1

            # The same weights are used for every single time step predictions
            # therefore the gradients must accumulate for all the parameters!

            # updates for the softmax classifier
            dl_dby += dl_dzpred
            dl_dway += np.dot(dl_dzpred, a.T)

            # this activation goes to the cell in the next time step and is used again
            # so it gets a gradient portion directly from the softmax classifier and from the next time step
            dl_da = np.dot(self.way.T, dl_dzpred) + dl_da_layer_ahead
            # backprop through the nonlinearity to get dl_dzactiv
            da_dzactiv = self.activation_function.getDerivative_wrtInput(
                z_activ)
            dl_dzactiv = dl_da * da_dzactiv

            dl_dba += dl_dzactiv
            dl_dwaa += np.dot(dl_dzactiv, a_prev.T)
            dl_dwax += np.dot(dl_dzactiv, x_t.T)

            # pass gradient back to prev activation
            dl_da_layer_ahead = np.dot(self.waa, dl_da)

        # RNN's can suffer from exploding gradients, so before we update the params
        # we are going to do gradient clipping
        params = [self.way, self.waa, self.wax, self.ba, self.by]
        dparams = [dl_dway, dl_dwaa, dl_dwax, dl_dba, dl_dby]
        gradientClipping(dparams)
        self.way, self.waa, self.wax, self.ba, self.by = optim.updateParams(
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
        for _ in range(totalGeneratingSteps):
            a, predicted_vector, _, _, _ = self._compute_one_step_forward(
                x_t, a_prev, temperature)
            # lets choose a letter weighted by the probability given by the
            # predicted vector
            predicted_index = np.random.choice(self.num_input_features,
                                               p=predicted_vector.ravel())
            # the predictiion from this timestep is the input for the next
            # time step
            x_t = predicted_vector
            a_prev = a
            # get the char that the index represents and join it to the output
            predicted_letter = idxToChar[predicted_index]
            output.append(predicted_letter)
        return "".join(output)

    def _initialize_weights(self, size1, size2):
        # sample from a standard normal distribution and multiple by 0.01
        # to init weights
        return np.random.randn(size1, size2) * 0.01

    def _initialize_bias(self, size1):
        return np.zeros((size1, 1))
