""" This module contains unit tests for the batch norm layer """
import unittest
import numpy as np
import tensorflow as tf
from machine_learning_algorithms.supervised_learning.classifiers.multi_layer_perceptron import MultiLayerPerceptron
from machine_learning_algorithms.neural_net_utility.neural_net_layers import DenseBatchNormLayer
from machine_learning_algorithms.supervised_learning.classifiers.softmax_regression import SoftmaxRegression
from machine_learning_algorithms.neural_net_utility.activation_functions import Softmax, ReLU
from machine_learning_algorithms.neural_net_utility.optimizer import GradientDescentMomentum, RMSProp
from machine_learning_algorithms.utility.score_functions import accuracy
from machine_learning_algorithms.utility.misc import one_hot_encode


class TestBatchNorm(unittest.TestCase):
    """ This class contains unit tests for the batch norm layer """

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
        self.saved_y = self.y_train.reshape(1, -1)
        self.y_train = one_hot_encode(self.y_train.reshape(1, -1))
        self.y_test = one_hot_encode(self.y_test.reshape(1, -1))
        self.x_mini_train = self.x_train[:, :1000].reshape(784, -1)
        self.y_mini_train = self.y_train[:, :1000]
        self.x_mini_valid = self.x_train[:, 1000:2000].reshape(784, -1)
        self.y_mini_valid = self.y_train[:, 1000:2000]

    def test_overfit_small_batch(self):
        multi_layer_perceptron = MultiLayerPerceptron(
            typeSupervised="multiclass", numberInputFeatures=784)

        multi_layer_perceptron.add_layer(num_neurons=100,
                                         activation_function=ReLU(),
                                         layer=DenseBatchNormLayer)
        multi_layer_perceptron.add_layer(num_neurons=10,
                                         activation_function=Softmax(),
                                         isSoftmax=True)

        train_loss1, train_acc1 = multi_layer_perceptron.fit(
            self.x_train[:, :100].reshape(784, -1),
            self.y_train[:, :100],
            num_epochs=150,
            ret_train_loss=True,
            optim=RMSProp(),
            learn_rate=0.001)
        predictions1 = multi_layer_perceptron.predict_multi_layer_perceptron(
            self.x_train[:, :100].reshape(784, -1))
        acc = accuracy(self.saved_y[:, :100].reshape(1, -1), predictions1)
        print(train_loss1)
        print(acc)
        self.assertLessEqual(train_loss1[-1], 0.09)
        self.assertEqual(acc, 1)

        multi_layer_perceptron2 = MultiLayerPerceptron(
            typeSupervised="multiclass", numberInputFeatures=784)

        multi_layer_perceptron2.add_layer(num_neurons=100,
                                          activation_function=ReLU(),
                                          layer=DenseBatchNormLayer)
        multi_layer_perceptron2.add_layer(num_neurons=10,
                                          activation_function=Softmax(),
                                          isSoftmax=True)

        train_loss2, train_acc2 = multi_layer_perceptron2.fit(
            self.x_train[:, :100].reshape(784, -1),
            self.y_train[:, :100],
            num_epochs=150,
            ret_train_loss=True,
            optim=GradientDescentMomentum(),
            learn_rate=0.1)
        predictions2 = multi_layer_perceptron2.predict_multi_layer_perceptron(
            self.x_train[:, :100].reshape(784, -1))
        acc2 = accuracy(self.saved_y[:, :100].reshape(1, -1), predictions2)
        print(train_loss2)
        print(acc2)
        self.assertLessEqual(train_loss2[-1], 0.09)
        self.assertEqual(acc2, 1)

    def test_bigger_batch_batchnorm(self):
        multi_layer_perceptron = MultiLayerPerceptron(
            typeSupervised="multiclass",
            numberInputFeatures=784,
            regularization="L2",
            reg_parameter=0.01)

        multi_layer_perceptron.add_layer(num_neurons=100,
                                         activation_function=ReLU(),
                                         layer=DenseBatchNormLayer)
        multi_layer_perceptron.add_layer(num_neurons=100,
                                         activation_function=ReLU(),
                                         layer=DenseBatchNormLayer)
        multi_layer_perceptron.add_layer(num_neurons=10,
                                         activation_function=Softmax(),
                                         isSoftmax=True)

        train_loss, valid_loss, train_acc, valid_acc = multi_layer_perceptron.fit(
            self.x_mini_train,
            self.y_mini_train,
            xvalid=self.x_mini_valid,
            yvalid=self.y_mini_valid,
            num_epochs=500,
            ret_train_loss=True,
            optim=GradientDescentMomentum(),
            learn_rate=0.1)
        print(train_loss)
        print("\n")
        print(valid_loss)
        print("\n")
        print(train_acc)
        print("\n")
        print(valid_acc)

    def test_bigger_batch_normal(self):
        multi_layer_perceptron = MultiLayerPerceptron(
            typeSupervised="multiclass", numberInputFeatures=784)

        multi_layer_perceptron.add_layer(num_neurons=100,
                                         activation_function=ReLU())
        multi_layer_perceptron.add_layer(num_neurons=100,
                                         activation_function=ReLU())
        multi_layer_perceptron.add_layer(num_neurons=10,
                                         activation_function=Softmax(),
                                         isSoftmax=True)

        train_loss, valid_loss, train_acc, valid_acc = multi_layer_perceptron.fit(
            self.x_mini_train,
            self.y_mini_train,
            xvalid=self.x_mini_valid,
            yvalid=self.y_mini_valid,
            num_epochs=100,
            ret_train_loss=True,
            optim=GradientDescentMomentum(),
            learn_rate=0.1)
        print(train_loss)
        print("\n")
        print(valid_loss)
        print("\n")
        print(train_acc)
        print("\n")
        print(valid_acc)


if __name__ == "__main__":
    unittest.main()