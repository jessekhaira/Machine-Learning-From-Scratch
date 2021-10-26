""" This module contains unit tests for a convolutional network """
import unittest
import numpy as np
import tensorflow as tf
from machine_learning_algorithms.neural_net_utility.neural_net_layers import DenseLayer
from machine_learning_algorithms.neural_net_utility.convolutional_layers import Conv2D
from machine_learning_algorithms.neural_net_utility.convolutional_layers import Pool
from machine_learning_algorithms.supervised_learning.classifiers.convolutional_neural_network import ConvolutionalNeuralNetwork
from machine_learning_algorithms.neural_net_utility.activation_functions import ReLU, Softmax
from machine_learning_algorithms.neural_net_utility.optimizer import GradientDescentMomentum, AdaGrad
from machine_learning_algorithms.utility.misc import oneHotEncode


class TestConvolutionalNet(unittest.TestCase):
    """ This class contains unit tests for a convolutional network """

    def setUp(self):
        (self.x_train,
         self.y_train), (self.x_test,
                         self.y_test) = tf.keras.datasets.mnist.load_data()
        self.x_train = np.array(self.x_train, dtype=float)
        self.x_test = np.array(self.x_test, dtype=float)
        self.x_train /= 255
        self.x_test /= 255
        self.x_train = self.x_train
        self.x_test = self.x_test
        self.saved_y = self.y_train.reshape(1, -1)
        self.y_train = oneHotEncode(self.y_train.reshape(1, -1))
        self.y_test = oneHotEncode(self.y_test.reshape(1, -1))
        self.x_mini_train = self.x_train[:1000, :, :].reshape(1000, 1, 28, 28)
        self.y_mini_train = self.y_train[:, :1000]
        self.x_mini_valid = self.x_train[:1000, :, :].reshape(1000, 1, 28, 28)
        self.y_mini_valid = self.y_train[:, 1000:2000]

        self.x_train = self.x_train.reshape(60000, 1, 28, 28)
        self.x_test = self.x_test.reshape(10000, 1, 28, 28)
        return super().setUp()

    def test_overfit_smallbatch(self):
        """ We're gonna test if we can overfit a small conv net on a small
        batch w/ just one layer followed by a softmax classifier.
        If we can't.. then somethings wrong with the backprop for the conv
        layer.
        """
        obj2 = ConvolutionalNeuralNetwork(typeSupervised="multiclass",
                                          inputDepth=1)
        paramsLayer1 = {
            "filterSize": 3,
            "inputDepth": None,
            "numFilters": 5,
            "activationFunction": ReLU(),
            "padding": "same",
            "stride": 1,
            "finalConvLayer": True
        }
        obj2.addConvNetLayer(Conv2D, **paramsLayer1)

        paramsLayer6 = {
            "num_neurons": 10,
            "activationFunction": Softmax(),
            "regularization": None,
            "reg_parameter": None,
            "isSoftmax": 1
        }
        obj2.addConvNetLayer(DenseLayer, **paramsLayer6)

        train_loss, train_acc = obj2.fit(self.x_mini_train[:32],
                                         self.y_mini_train[:, :32],
                                         num_epochs=350,
                                         ret_train_loss=True,
                                         verbose=True,
                                         learn_rate=0.4,
                                         optim=GradientDescentMomentum())
        print(train_loss)

        # Looks good!

    def test_overfit_smallbatch_mediumnet(self):
        """ Testing conv layer followed by avg pool layer
        followed by classifier
        """
        obj3 = ConvolutionalNeuralNetwork(typeSupervised="multiclass",
                                          inputDepth=1)
        paramsLayer1 = {
            "filterSize": 3,
            "inputDepth": None,
            "numFilters": 5,
            "activationFunction": ReLU(),
            "padding": "same",
            "stride": 1,
            "finalConvLayer": False
        }
        obj3.addConvNetLayer(Conv2D, **paramsLayer1)

        paramsLayer2 = {
            "filterSize": 3,
            "stride": 2,
            "finalConvLayer": True,
            "poolType": "avg",
            "padding": "valid"
        }
        obj3.addConvNetLayer(Pool, **paramsLayer2)

        paramsLayer6 = {
            "num_neurons": 10,
            "activationFunction": Softmax(),
            "regularization": None,
            "reg_parameter": None,
            "isSoftmax": 1
        }
        obj3.addConvNetLayer(DenseLayer, **paramsLayer6)

        train_loss, train_acc = obj3.fit(self.x_mini_train[:32],
                                         self.y_mini_train[:, :32],
                                         num_epochs=500,
                                         ret_train_loss=True,
                                         verbose=True,
                                         learn_rate=0.4,
                                         optim=GradientDescentMomentum())

        # Looks good!

    def testOverfitSmallBatch_medNetMaxPool(self):
        # testing conv layer followed by avg pool layer followed by classifier
        obj3 = ConvolutionalNeuralNetwork(typeSupervised="multiclass",
                                          inputDepth=1)
        paramsLayer1 = {
            "filterSize": 3,
            "inputDepth": None,
            "numFilters": 5,
            "activationFunction": ReLU(),
            "padding": "same",
            "stride": 1,
            "finalConvLayer": False
        }
        obj3.addConvNetLayer(Conv2D, **paramsLayer1)

        paramsLayer2 = {
            "filterSize": 3,
            "stride": 2,
            "finalConvLayer": True,
            "poolType": "max",
            "padding": "valid"
        }
        obj3.addConvNetLayer(Pool, **paramsLayer2)

        paramsLayer6 = {
            "num_neurons": 10,
            "activationFunction": Softmax(),
            "regularization": None,
            "reg_parameter": None,
            "isSoftmax": 1
        }
        obj3.addConvNetLayer(DenseLayer, **paramsLayer6)

        train_loss, train_acc = obj3.fit(self.x_mini_train[:32],
                                         self.y_mini_train[:, :32],
                                         num_epochs=500,
                                         ret_train_loss=True,
                                         verbose=True,
                                         learn_rate=0.4,
                                         optim=GradientDescentMomentum())

        # trains fine but loss goes down way slower than average pooling since we only send the gradient through the maximum neuron
        # versus every single neuron. Doesn't make a huge difference, just need to train for way longer.

    def testFullNet(self):
        # train the full net on 60k images - takes a long time to train but gets great performance!
        obj1 = ConvolutionalNeuralNetwork(typeSupervised="multiclass",
                                          inputDepth=1)

        paramsLayer1 = {
            "filterSize": 3,
            "inputDepth": None,
            "numFilters": 5,
            "activationFunction": ReLU(),
            "padding": "same",
            "stride": 1,
            "finalConvLayer": False
        }
        obj1.addConvNetLayer(Conv2D, **paramsLayer1)
        paramsLayer2 = {
            "filterSize": 3,
            "stride": 2,
            "finalConvLayer": False,
            "poolType": "avg",
            "padding": "valid"
        }
        obj1.addConvNetLayer(Pool, **paramsLayer2)

        paramsLayer3 = {
            "filterSize": 3,
            "inputDepth": None,
            "numFilters": 5,
            "activationFunction": ReLU(),
            "padding": "same",
            "stride": 1,
            "finalConvLayer": False
        }
        obj1.addConvNetLayer(Conv2D, **paramsLayer3)
        paramsLayer4 = {
            "filterSize": 3,
            "finalConvLayer": True,
            "stride": 2,
            "finalConvLayer": True,
            "poolType": "avg",
            "padding": "valid"
        }
        obj1.addConvNetLayer(Pool, **paramsLayer4)

        paramsLayer5 = {
            "num_neurons": 75,
            "activationFunction": ReLU(),
            "regularization": None,
            "reg_parameter": None
        }
        obj1.addConvNetLayer(DenseLayer, **paramsLayer5)

        paramsLayer6 = {
            "num_neurons": 10,
            "activationFunction": Softmax(),
            "regularization": None,
            "reg_parameter": None,
            "isSoftmax": 1
        }
        obj1.addConvNetLayer(DenseLayer, **paramsLayer6)
        train_loss, train_acc = obj1.fit(self.x_train,
                                         self.y_train,
                                         xvalid=self.x_test,
                                         yvalid=self.y_test,
                                         num_epochs=150,
                                         ret_train_loss=True,
                                         verbose=True,
                                         learn_rate=0.01,
                                         batch_size=128,
                                         optim=AdaGrad())


if __name__ == "__main__":
    unittest.main()
