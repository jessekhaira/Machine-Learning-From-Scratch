import numpy as np
from machine_learning_algorithms.neural_net_utility.neural_net_layers import DenseLayer, DenseBatchNormLayer
from machine_learning_algorithms.neural_net_utility.activation_functions import ReLU, IdentityActivation, TanH, Sigmoid
from machine_learning_algorithms.neural_net_utility.loss_functions import mean_squared_error
from machine_learning_algorithms.neural_net_utility.neural_net_base import NeuralNetworkBase
from machine_learning_algorithms.neural_net_utility.optimizer import GradientDescent, GradientDescentMomentum, RMSProp, Adam
import matplotlib.pyplot as plt


class Deep_Autoencoder(NeuralNetworkBase):
    """
    This class represents a Deep Autoencoder made for MNIST. An autoencoder is a type of neural network used in unsupervised learning,
    typically to learn data codings for dimensionality reduction. 

    The Autoencoder consists of two fully connected nets: An encoder and a decoder. The encoder takes the high dimensional
    input and encodes it in a size_encoding dense layer. From the size_encoding dense layer, the original input is reconstructed
    by the decoder.

    The whole architecture is then trained with Mean Squared Loss, with the objective obviously being to reconstruct the original
    inputs from the low dimensional encoding.

    Parameters:
    -> size_encoding (int): Integer representing the size of the encoding the autoencoder will encode to. 
    """

    def __init__(self, size_encoding):
        self.size_encoding = size_encoding
        self.num_features = 784
        # Design decision - encoder and decoder are symmetric
        # not a hard constraint but its convienant as it removes a hyperparameter
        self.img_height = 28
        self.img_width = 28
        self.encoder = self._buildEncoder()
        self.decoder = self._buildDecoder()
        # now stack the decoder layers right after the encoder layers
        self.layers = self.encoder + self.decoder
        self.lossFunction = mean_squared_error()

        # after we've defined the architecture, the autoencoder is like any other
        # neural net

    def _buildEncoder(self):
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

    def _buildDecoder(self):
        decoder = []
        decoder.append(
            DenseBatchNormLayer(num_in=self.size_encoding,
                                num_layer=150,
                                activationFunction=ReLU()))
        decoder.append(
            DenseBatchNormLayer(num_in=150,
                                num_layer=300,
                                activationFunction=ReLU()))
        # decoded outputs - sigmoid activation used because inputs are in the range of 0 -1
        # so our outputs should also be in between the range of 0-1
        decoder.append(
            DenseLayer(num_in=300,
                       num_layer=self.num_features,
                       activationFunction=Sigmoid()))
        return decoder

    def fit(self,
            xtrain,
            num_epochs=10,
            batch_size=32,
            ret_train_loss=False,
            learn_rate=0.1,
            optim=GradientDescent(),
            verbose=False):
        num_batches = xtrain.shape[1] // batch_size
        train_loss = []
        for epoch in range(num_epochs):
            currStart = 0
            currEnd = batch_size
            lossEpoch = []
            for i in range(num_batches):
                if verbose:
                    print("epoch num %s, batch num %s" % (epoch, i))
                curr_x = xtrain[:, currStart:currEnd]
                currStart = currEnd
                currEnd += batch_size
                pred_miniBatch = self._forward_propagate(curr_x)
                loss = self._calculateLoss(curr_x, pred_miniBatch, self.layers)
                lossEpoch.append(loss)
                backpropInit = self.lossFunction.derivativeLoss_wrtPrediction(
                    curr_x, pred_miniBatch)
                self._backward_propagate(backpropInit, learn_rate, optim, epoch,
                                         curr_x, curr_x)

            train_loss.append(np.mean(lossEpoch))

            # provide updates during training for sanitys sake
            if verbose and epoch % 100 == 0:
                print("Finished epoch %s" % (epoch))
                print("Train loss: %s" % (train_loss[-1]))
                self.imgs_checkpoints(epoch, xtrain)
        if ret_train_loss:
            return train_loss

    def imgs_checkpoints(self, epoch, X):
        """
        This method generates a batch of images from the autoencoder, plots them,
        and saves the figure produced at various training checkpoints to assess how well 
        the autoencoder is doing at reconstructing the inputs.
        
        Parameters:
        -> epoch (int): The epoch at which the imgs are generated
        -> X (NumPy matrix): Matrix of training examples
        
        Returns:
        None
        """
        # we are going to generate 9 images
        # and then plot 9 images on subplots
        num_rows, num_cols = 3, 3
        totalPics = 9
        #random batch of 9 pics w/o replacement
        pic_axes = np.random.choice(X.shape[1], size=totalPics, replace=False)
        train_9 = X[:, pic_axes]
        # plots
        fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols)
        # get generated imgs
        imgs = self.predict(train_9,
                            supervised=False).reshape(-1, self.img_height,
                                                      self.img_width)
        imgIdx = 0
        for i in range(num_rows):
            for j in range(num_cols):
                ax[i, j].imshow(imgs[imgIdx, :, :], cmap='gray')
                # we just want to see a grid of pictures, so we turn the axes off
                ax[i, j].axis('off')
                imgIdx += 1
        # save the figures!
        fig.savefig("AutoEncoder Imgs Epoch Num %s" % epoch)
        plt.close()
