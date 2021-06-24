""" This module contains code for the restricted boltzmann
machine learning algorithm """
import numpy as np
from ML_algorithms.neural_net_utility.activation_functions import Sigmoid
import matplotlib.pyplot as plt


class RBM(object):
    """ This class represents a bernoulli Restricted Boltzmann Machine (RBM). A
    RBM is a generative neural network used to learn a probability distribution
    over its set of inputs. An RBM is restricted, meaning that no nodes within
    the same layer are connected, but a node in a given layer is fully connected
    connected to every node in an adjacent layer. These connections are
    symmetric between the hidden nodes and visible nodes.

    RBMs have found applications in dimensionality reduction, collaborative
    filtering, classification, and many other areas.

    Defaults are set as reccomended by:
        http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf

    Attributes:
        n_visible:
            The number of features in a single example being fed into the net

        n_hidden:
            The number of features being learned during training

        batch_size:
            The size of a mini-batch during training

        n_epochs:
            Number of epochs to train for

        ret_train:
            Boolean indicating whether to return the errors while training and
            return images sampled during training

        seed:
            Integer representing random seed to use

        k:
            Integer representing number of iterations to perform with gibbs
            sampling

        is_img:
            Boolean value indicating whether the RBM is being used for images
            for visualization purposes during training

        img_h:
            Integer indicating height of image input, if image

        img_w:
            Integer indicating width of image input, if image

        img_d:
            Int indicating depth of image input, if image
    """

    def __init__(self,
                 n_visible: int,
                 n_hidden: int = 128,
                 learning_rate: int = 0.1,
                 batch_size: int = 100,
                 n_epochs: int = 100,
                 ret_train: int = True,
                 k: int = 1,
                 weight_decay: int = 0,
                 is_img: bool = False,
                 img_h: int = -1,
                 img_w: int = -1,
                 img_d: int = -1,
                 seed: int = 9):
        np.random.seed(seed)
        self.seed = seed
        self.w, self.b_v, self.b_h = self._init_weights(n_visible, n_hidden)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.sigmoid = Sigmoid()
        self.weight_decay = weight_decay
        self.k = k
        # optional but useful when inputs are images
        self.img_h = img_h
        self.img_w = img_w
        self.img_d = img_d
        self.is_img = is_img
        self.ret_train = ret_train

    def _init_weights(self, n_visible, n_hidden):
        # initialize weights to small random values chosen from a zero-mean
        # Gaussian distribution w/ std dev of 0.01
        w = np.random.normal(0, 0.01, (n_visible, n_hidden))
        # init biases to zeros
        b_v = np.zeros((1, n_visible))
        b_h = np.zeros((1, n_hidden))
        return w, b_v, b_h

    def _get_reconstruct_error(self, real, reconstruct):
        # avg squared error of real x and reconstructed x from the hidden units
        return np.mean((real - reconstruct)**2)

    def v_to_h(self, vis):
        hidden_probabilities = self.sigmoid.compute_output(
            vis.dot(self.w) + self.b_h)

        # In the training guide, turning the hidden unit activations to either
        # 0 or 1 (IE: 1 bit) is described as a strong regularizer as the hidden
        # units will be unable to communicate a real value to the visible units
        # during reconstruction
        sampled_h = self._sample(hidden_probabilities)

        return hidden_probabilities, sampled_h

    def h_to_v(self, sampled_h):
        # assuming the visible units use the logistic function - use real-valued
        # probabilities for both the data and the reconstructions
        x_reconstruct = self.sigmoid.compute_output(
            sampled_h.dot(self.w.T) + self.b_v)
        return x_reconstruct

    def train(self,
              data: np.ndarray,
              verbose: bool = True,
              sampling_epochs: int = 10):
        """This method is used to train the RBM on data of shape (M, n_visible),
        where M is the number of examples in the dataset and n_visible is the
        number of features.

        As this is a bernoulli RBM, the input values should be preprocessed to
        be between 0 and 1.

        Args:
            data:
                Numpy Matrix of shape (M,N) where M is the number of examples
                and N is the number of features
            verbose:
                Boolean indicating whether or not to provide updates during
                training
            sampling_epochs:
                If verbose, the number of epochs to wait before producing
                sampled images and reconstruction errors
        """
        reconstruction_error = []
        reconstructed_vectors = []

        # normal neural net training boilerplate code
        for epoch in range(self.n_epochs):
            num_batches = data.shape[0] // self.n_epochs
            batch_iterator = 0
            epoch_errors = []
            for _ in range(num_batches):
                x_orig = data[batch_iterator:batch_iterator +
                              self.batch_size, :]
                batch_iterator += self.batch_size

                hidden_probabilities, sampled_h = self.v_to_h(x_orig)
                # use the real probabilites for the positive gradient
                pos_gradient = x_orig.T.dot(hidden_probabilities)

                # Reconstruction -> using the same weight values, which is why
                # this forms an undirected graph

                # we do k steps of alternating gibbs sampling in order to get
                # second term for weight update makes the RBM learn better
                for _ in range(self.k):
                    # use sampled h with some activations turned to 0, others
                    # turned to 1 -> hidden neurons only communicating one bit
                    # to visible neurons
                    x_reconstruct = self.h_to_v(sampled_h)
                    h_reconstruct_prob, sampled_h = self.v_to_h(x_reconstruct)

                # get negative gradient
                neg_gradient = x_reconstruct.T.dot(h_reconstruct_prob)

                # Paper says its helpful to divide total gradient computed on a
                # mini-batch by the size of the mini-batch

                #regularization term included in the gradient for the function
                # if reg penalty specified
                self.w += self.learning_rate / self.batch_size * (
                    pos_gradient - neg_gradient -
                    (2 * self.weight_decay * self.w))
                self.b_h += self.learning_rate / self.batch_size * (
                    hidden_probabilities.sum(axis=0).reshape(1, -1) -
                    h_reconstruct_prob.sum(axis=0).reshape(1, -1))
                self.b_v += self.learning_rate / self.batch_size * (
                    x_orig.sum(axis=0).reshape(1, -1) -
                    x_reconstruct.sum(axis=0).reshape(1, -1))

                # reconstruction error
                epoch_errors.append(
                    self._get_reconstruct_error(x_orig, x_reconstruct))

            # add avg epoch error
            reconstruction_error.append(np.mean(epoch_errors))

            # reconstruct a random vector
            # don't want the same idx every time so we change the seed and then
            # sample then change it to default
            np.random.seed(epoch * 150)
            random_idx = np.random.randint(0, x_orig.shape[0])
            np.random.seed(self.seed)
            random_vector = data[random_idx, :]
            reconstructed_vectors.append(self.reconstruct(random_vector))

            if verbose and epoch % sampling_epochs == 0:
                print('Reconstruction error at epoch %s is %s' %
                      (epoch, reconstruction_error[-1]))
                if self.is_img:
                    self._show_imgs(epoch, random_vector,
                                    reconstructed_vectors[-1])

        if self.ret_train:
            return reconstruction_error, reconstructed_vectors

    def _show_imgs(self, epoch, *args):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
        for img, ax in zip(args, [ax1, ax2]):
            img = img.reshape(self.img_h, self.img_w,
                              self.img_d) if self.img_d != 1 else img.reshape(
                                  self.img_h, self.img_w)
            ax.imshow(img)
            ax.axis('off')

        fig.savefig('Restricted Boltzmann Machine Imgs Epoch Num %s' % epoch)
        plt.close()

    def _sample(self, prob_matrix):
        # from training guide - we want to make the hidden states binary 0 or 1
        # So a hidden unit turns on if the probability is greater than a
        # random number uniformly distributed between 0 and 1
        return (prob_matrix > np.random.rand(*prob_matrix.shape)).astype(
            np.float64)

    def reconstruct(self, x_orig):
        hidden_probabilities = self.sigmoid.compute_output(
            x_orig.dot(self.w) + self.b_h)
        sampled_h = self._sample(hidden_probabilities)
        return self.sigmoid.compute_output(sampled_h.dot(self.w.T) + self.b_v)
