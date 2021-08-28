""" This module contains code representing the unsupervised
machine learning algorithm the autoencoder """
import numpy as np
from machine_learning_algorithms.neural_net_utility.neural_net_layers import DenseLayer, DenseBatchNormLayer
from machine_learning_algorithms.neural_net_utility.activation_functions import ReLU, Sigmoid
from machine_learning_algorithms.neural_net_utility.loss_functions import mean_squared_error
from machine_learning_algorithms.neural_net_utility.neural_net_base import NeuralNetworkBase
from machine_learning_algorithms.neural_net_utility.optimizer import GradientDescent, Optimizer
import matplotlib.pyplot as plt


class AutoEncoder(NeuralNetworkBase):
    """ This class represents a Deep Autoencoder made for MNIST.
    An autoencoder is a type of neural network used in unsupervised
    learning, typically to perform dimensionality reduction.

    The Autoencoder consists of two fully connected nets: An encoder and
    a decoder. The encoder takes as input high dimensional vectors and
    embeds them in a lower dimension. From the embedding the original
    input is reconstructed by the decoder.

    The whole architecture is then trained with Mean Squared Loss, with
    the objective obviously being to reconstruct the original inputs
    from the low dimensional encoding.

    Attributes:
        size_encoding:
            Integer representing the size of the encoding the autoencoder
            will encode to

        num_input_features:
            Integer representing the number of dimensions present in the
            input vectors to the auto encoder

        img_height:
            Integer representing the height of the images shown during training

        img_width:
            Integer representing the width of the images shown during training

        encoder:
            Fully connected neural network representing the encoder portion
            of the autoencoder

        decoder:
            Fully connected neural network representing the decoder portion
            of the autoencoder

        layers:
            A list containing the two layers of the autoencoder

        objective_function:
            Function representing the objective function to use during
            training
    """

    def __init__(self,
                 size_encoding: int,
                 num_input_features: int = 784,
                 img_height: int = 28,
                 img_width: int = 28,
                 objective_function=mean_squared_error()):
        self.size_encoding = size_encoding
        self.num_input_features = num_input_features
        # Design decision - encoder and decoder are symmetric
        # not a hard constraint but its convienant as it removes
        # a hyperparameter
        self.img_height = img_height
        self.img_width = img_width
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        # now stack the decoder layers right after the encoder layers
        self.layers = self.encoder + self.decoder
        self.objective_function = objective_function

    def _build_encoder(self):
        encoder = []
        encoder.append(
            DenseBatchNormLayer(num_in=784,
                                num_layer=300,
                                activationFunction=ReLU()))
        encoder.append(
            DenseBatchNormLayer(num_in=300,
                                num_layer=150,
                                activationFunction=ReLU()))
        # fully encoded units
        encoder.append(
            DenseBatchNormLayer(num_in=150,
                                num_layer=self.size_encoding,
                                activationFunction=ReLU()))
        return encoder

    def _build_decoder(self):
        decoder = []
        decoder.append(
            DenseBatchNormLayer(num_in=self.size_encoding,
                                num_layer=150,
                                activationFunction=ReLU()))
        decoder.append(
            DenseBatchNormLayer(num_in=150,
                                num_layer=300,
                                activationFunction=ReLU()))
        # decoded outputs - sigmoid activation used because
        # inputs are in the range of 0 -1, so our outputs should
        # also be in between the range of 0-1
        decoder.append(
            DenseLayer(num_in=300,
                       num_layer=self.num_input_features,
                       activationFunction=Sigmoid()))
        return decoder

    def fit(self,
            xtrain: np.ndarray,
            num_epochs: int = 10,
            batch_size: int = 32,
            ret_train_loss: bool = False,
            learn_rate: float = 0.1,
            optim: Optimizer = GradientDescent(),
            verbose: bool = False):
        num_batches = xtrain.shape[1] // batch_size
        train_loss = []
        for epoch in range(num_epochs):
            curr_start = 0
            curr_end = batch_size
            loss_epoch = []
            for i in range(num_batches):
                if verbose:
                    print("epoch num %s, batch num %s" % (epoch, i))
                curr_x = xtrain[:, curr_start:curr_end]
                curr_start = curr_end
                curr_end += batch_size
                pred_mini_batch = self._forward_propagate(curr_x)
                loss = self._calculateLoss(curr_x, pred_mini_batch, self.layers)
                loss_epoch.append(loss)
                backprop_init = self.objective_function.get_gradient_pred(
                    curr_x, pred_mini_batch)
                self._backward_propagate(backprop_init, learn_rate, optim,
                                         epoch, curr_x, curr_x)

            train_loss.append(np.mean(loss_epoch))

            # provide updates during training for sanitys sake
            if verbose and epoch % 100 == 0:
                print("Finished epoch %s" % (epoch))
                print("Train loss: %s" % (train_loss[-1]))
                self.imgs_checkpoints(epoch, xtrain)
        if ret_train_loss:
            return train_loss

    def imgs_checkpoints(self, epoch: int, x: np.ndarray) -> None:
        """ This method generates a batch of images from the
        autoencoder, plots them, and saves the figure produced
        at various training checkpoints to assess how well the
        autoencoder is doing at reconstructing the inputs.

        Args:
            epoch:
                Integer representing the epoch at which the imgs are generated

            x:
                Numpy array of shape (num_examples, num_features) of
                training examples
        """
        # we are going to generate 9 images
        # and then plot 9 images on subplots
        num_rows, num_cols = 3, 3
        total_pics = 9
        #random batch of 9 pics w/o replacement
        pic_axes = np.random.choice(x.shape[1], size=total_pics, replace=False)
        train_9 = x[:, pic_axes]
        # plots
        fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols)
        # get generated imgs
        imgs = self.predict(train_9,
                            supervised=False).reshape(-1, self.img_height,
                                                      self.img_width)
        img_idx = 0
        for i in range(num_rows):
            for j in range(num_cols):
                ax[i, j].imshow(imgs[img_idx, :, :], cmap="gray")
                # we just want to see a grid of pictures, so we turn
                # the axes off
                ax[i, j].axis("off")
                img_idx += 1
        # save the figures!
        fig.savefig("AutoEncoder Imgs Epoch Num %s" % epoch)
        plt.close()
