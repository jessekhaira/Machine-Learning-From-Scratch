import unittest
import numpy as np
import tensorflow as tf
from machine_learning_algorithms.supervised_learning.classifiers.multi_layer_perceptron import MultiLayerPerceptron
from machine_learning_algorithms.neural_net_utility.neural_net_layers import DenseDropOutLayer, DenseLayer
from machine_learning_algorithms.neural_net_utility.activation_functions import ReLU, Softmax
from machine_learning_algorithms.neural_net_utility.optimizer import RMSProp
from machine_learning_algorithms.utility.misc import oneHotEncode

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.array(x_train, dtype=float)
x_test = np.array(x_test, dtype=float)
x_train /= 255
x_test /= 255
x_train = x_train.reshape(784, -1)
x_test = x_test.reshape(784, -1)
saved_y = y_train.reshape(1, -1)
y_train = oneHotEncode(y_train.reshape(1, -1))
y_test = oneHotEncode(y_test.reshape(1, -1))
print(y_train.shape)
print(y_test.shape)
print(x_train[:, :5])

x_miniTrain = x_train[:, :1000].reshape(784, -1)
y_miniTrain = y_train[:, :1000]
x_miniValid = x_train[:, 1000:2000].reshape(784, -1)
y_miniValid = y_train[:, 1000:2000]


class testMNIST_DropoutLayers(unittest.TestCase):

    def testLowKeepProb(self):
        MLP = MultiLayerPerceptron(typeSupervised="multiclass",
                                   numberInputFeatures=784)

        # make sure we get high training loss, low acccuracy when we dropout 99% of activations
        # just sanity checking the implementation of the droput layer
        MLP.add_layer(num_neurons=100,
                      activation_function=ReLU(),
                      layer=DenseDropOutLayer,
                      keep_prob=0.01)
        MLP.add_layer(num_neurons=10,
                      activation_function=Softmax(),
                      isSoftmax=True)
        print(isinstance(MLP.layers[0], DenseLayer))

        train_loss1, train_acc1 = MLP.fit(x_train[:, :100].reshape(784, -1),
                                          y_train[:, :100],
                                          num_epochs=500,
                                          ret_train_loss=True,
                                          optim=RMSProp(),
                                          learn_rate=0.001)
        print(train_loss1)
        print(train_acc1)

        ## predictions basically don't change from 0.1 for each class for every example so the loss for every epoch is
        ## -ln(0.1) = 2.302, which makes sense because the regularization is so strong that the network isn't learning anything

    def testOverFitSmallBatch(self):
        MLP = MultiLayerPerceptron(typeSupervised="multiclass",
                                   numberInputFeatures=784)

        # make sure we get high training loss, low acccuracy when we dropout 99% of activations
        # just sanity checking the implementation of the droput layer
        MLP.add_layer(num_neurons=100,
                      activation_function=ReLU(),
                      layer=DenseDropOutLayer,
                      keep_prob=0.6)
        MLP.add_layer(num_neurons=10,
                      activation_function=Softmax(),
                      isSoftmax=True)

        train_loss1, train_acc1 = MLP.fit(x_train[:, :100].reshape(784, -1),
                                          y_train[:, :100],
                                          num_epochs=500,
                                          ret_train_loss=True,
                                          optim=RMSProp(),
                                          learn_rate=0.001)
        print(train_loss1)
        print(train_acc1)

    #     # With a reasonable dropout probability, we can overfit to a small batch of data so it looks like everything is wired correctly

    def testFitBigBatch(self):
        MLP = MultiLayerPerceptron(typeSupervised="multiclass",
                                   numberInputFeatures=784)
        MLP.add_layer(num_neurons=25,
                      activation_function=ReLU(),
                      layer=DenseDropOutLayer,
                      keep_prob=0.09)
        MLP.add_layer(num_neurons=25,
                      activation_function=ReLU(),
                      layer=DenseDropOutLayer,
                      keep_prob=0.09)
        MLP.add_layer(num_neurons=10,
                      activation_function=Softmax(),
                      isSoftmax=True)

        train_loss1, valid_loss, train_acc, valid_acc = MLP.fit(
            x_miniTrain,
            y_miniTrain,
            x_miniValid,
            y_miniValid,
            num_epochs=800,
            ret_train_loss=True,
            optim=RMSProp(),
            learn_rate=0.001)
        print(train_loss1)
        print('\n')
        print(valid_loss)
        print('\n')
        print(train_acc)
        print('\n')
        print(valid_acc)

        # The architecture goes between overfitting to the training set when keep_prob is low, to underfitting when keep prob is high.
        # So overall we could remedy the overfitting by training on more examples and adding L2 regularization
        # But the dropout layer itself seems to be implemented fine


if __name__ == "__main__":
    unittest.main()