""" This module contains unit tests for dropout layers """
import unittest
import numpy as np
import tensorflow as tf
from machine_learning_algorithms.supervised_learning.classifiers.multi_layer_perceptron import MultiLayerPerceptron
from machine_learning_algorithms.neural_net_utility.neural_net_layers import DenseDropOutLayer
from machine_learning_algorithms.neural_net_utility.activation_functions import ReLU, Softmax
from machine_learning_algorithms.neural_net_utility.optimizer import RMSProp
from machine_learning_algorithms.utility.misc import oneHotEncode


class TestDropoutLayers(unittest.TestCase):
    """ This class contains unit tests for dropout layers """

    def setUp(self):
        (self.x_train,
         self.y_train), (self.x_test,
                         self.y_test) = tf.keras.datasets.mnist.load_data()
        self.x_train = np.array(self.x_train, dtype=float)
        self.x_test = np.array(self.x_test, dtype=float)
        self.x_train /= 255
        self.x_test /= 255
        self.x_train = self.x_train.reshape(784, -1)
        self.x_test = self.x_test.reshape(784, -1)
        self.y_train = oneHotEncode(self.y_train.reshape(1, -1))
        self.y_test = oneHotEncode(self.y_test.reshape(1, -1))
        self.x_mini_train = self.x_train[:, :1000].reshape(784, -1)
        self.y_mini_train = self.y_train[:, :1000]
        self.x_mini_valid = self.x_train[:, 1000:2000].reshape(784, -1)
        self.y_mini_valid = self.y_train[:, 1000:2000]
        return super().setUp()

    def test_1(self):
        multi_layer_perceptron = MultiLayerPerceptron(
            typeSupervised="multiclass", numberInputFeatures=784)

        # make sure we get high training loss, low acccuracy when we
        # dropout 99% of activations. Just sanity checking the implementation
        # of the droput layer
        multi_layer_perceptron.add_layer(num_neurons=100,
                                         activation_function=ReLU(),
                                         layer=DenseDropOutLayer,
                                         keep_prob=0.01)
        multi_layer_perceptron.add_layer(num_neurons=10,
                                         activation_function=Softmax(),
                                         isSoftmax=True)

        train_loss1, train_acc1 = multi_layer_perceptron.fit(
            self.x_train[:, :100].reshape(784, -1),
            self.y_train[:, :100],
            num_epochs=500,
            ret_train_loss=True,
            optim=RMSProp(),
            learn_rate=0.001)
        train_acc1 = np.average(train_acc1)
        self.assertLessEqual(train_acc1, 0.40)

    def test_2(self):
        """ With a reasonable dropout probability, we can overfit to a small
        batch of data so it looks like everything is wired correctly
        """

        multi_layer_perceptron = MultiLayerPerceptron(
            typeSupervised="multiclass", numberInputFeatures=784)

        # make sure we get high training loss, low acccuracy when we dropout
        # 99% of activations. Just sanity checking the implementation of the
        # dropout layer
        multi_layer_perceptron.add_layer(num_neurons=100,
                                         activation_function=ReLU(),
                                         layer=DenseDropOutLayer,
                                         keep_prob=0.6)
        multi_layer_perceptron.add_layer(num_neurons=10,
                                         activation_function=Softmax(),
                                         isSoftmax=True)

        train_loss1, train_acc1 = multi_layer_perceptron.fit(
            self.x_train[:, :100].reshape(784, -1),
            self.y_train[:, :100],
            num_epochs=500,
            ret_train_loss=True,
            optim=RMSProp(),
            learn_rate=0.001)
        train_acc1 = np.average(train_acc1)
        self.assertGreaterEqual(train_acc1, 0.89)

    def test_3(self):
        """ The architecture goes between overfitting to the training set
        when keep_prob is low, to underfitting when keep prob is high.
        So overall we could remedy the overfitting by training on more examples
        and adding L2 regularization.

        But the dropout layer itself seems to be implemented fine
        """

        multi_layer_perceptron = MultiLayerPerceptron(
            typeSupervised="multiclass", numberInputFeatures=784)
        multi_layer_perceptron.add_layer(num_neurons=25,
                                         activation_function=ReLU(),
                                         layer=DenseDropOutLayer,
                                         keep_prob=0.09)
        multi_layer_perceptron.add_layer(num_neurons=25,
                                         activation_function=ReLU(),
                                         layer=DenseDropOutLayer,
                                         keep_prob=0.09)
        multi_layer_perceptron.add_layer(num_neurons=10,
                                         activation_function=Softmax(),
                                         isSoftmax=True)

        train_loss1, valid_loss, train_acc, valid_acc = (
            multi_layer_perceptron.fit(self.x_mini_train,
                                       self.y_mini_train,
                                       self.x_mini_valid,
                                       self.y_mini_valid,
                                       num_epochs=800,
                                       ret_train_loss=True,
                                       optim=RMSProp(),
                                       learn_rate=0.001))

        self.assertGreaterEqual(train_acc, 0.85)
        self.assertGreaterEqual(valid_acc, 0.80)

        print(train_loss1)
        print("\n")
        print(valid_loss)
        print("\n")
        print(train_acc)
        print("\n")
        print(valid_acc)


if __name__ == "__main__":
    unittest.main()