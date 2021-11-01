""" This module contains unit tests for the multi layer perceptron
algorithm """
import unittest
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from machine_learning_algorithms.utility.ScoreFunctions import accuracy
from machine_learning_algorithms.utility.misc import oneHotEncode
from machine_learning_algorithms.supervised_learning.classifiers.multi_layer_perceptron import MultiLayerPerceptron
from machine_learning_algorithms.supervised_learning.classifiers.softmax_regression import SoftmaxRegression
from machine_learning_algorithms.neural_net_utility.activation_functions import ReLU, TanH, Sigmoid, Softmax


def create_spiral_dataset():
    # Spiral Dataset from CS231N
    N = 100    # number of points per class
    D = 2    # dimensionality
    K = 3    # number of classes
    x = np.zeros((N * K, D))    # data matrix (each row = single example)
    y = np.zeros(N * K, dtype="uint8")    # class labels
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)    # radius
        t = np.linspace(j * 4,
                        (j + 1) * 4, N) + np.random.randn(N) * 0.2    # theta
        x[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    x = x.T
    notEncodedy = y
    y = oneHotEncode(y.reshape(1, -1))
    return x, y, notEncodedy


class TestMultiLayerPerceptron(unittest.TestCase):

    def setUp(self) -> None:
        self.x, self.y = load_breast_cancer(return_X_y=True)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.15, random_state=42)
        self.x_train = preprocessing.scale(self.x_train)
        self.x_test = preprocessing.scale(self.x_test).T
        self.y_test = self.y_test.T.reshape(1, -1)

        self.x_train, self.x_valid, self.y_train, self.y_valid = (
            train_test_split(self.x_train,
                             self.y_train,
                             test_size=0.15,
                             random_state=42))

        self.x_train = self.x_train.T
        self.x_valid = self.x_valid.T
        self.y_train = self.y_train.T.reshape(1, -1)
        self.y_valid = self.y_valid.T.reshape(1, -1)

        return super().setUp()

    def test_binaryclassification_unregularized(self):
        """ Learning rate KEY: If its set to high, you will diverge. If its set
        to low, your network won"t learn anything. The more layers you have,
        the higher the learning rate should be. Example seen here:
        1 layer only - learn rate 0.1 suffices. With
        two layers ~ 0.7, with 3 layers ~2.5. """
        MLP = MultiLayerPerceptron(typeSupervised="binary",
                                   numberInputFeatures=self.x_train.shape[0])
        # Just add Dense Layers
        # 10 features learnt in first hidden layer w/ ReLU
        MLP.add_layer(10, activation_function=ReLU())
        # # 5 features learnt in second hidden layer w/ ReLU
        MLP.add_layer(5, activation_function=ReLU())
        # # Output layer sigmoid activation
        MLP.add_layer(1, activation_function=Sigmoid())

        train_loss, valid_loss, train_acc, valid_acc = MLP.fit(
            self.x_train,
            self.y_train,
            self.x_valid,
            self.y_valid,
            ret_train_loss=True,
            num_epochs=100,
            learn_rate=2.6)
        preds = MLP.predict_multi_layer_perceptron(self.x_test, 0.5)
        acc = accuracy(self.y_test, preds)
        print(acc)
        print(train_loss)
        print("\n")
        print(valid_loss)

        # Can get 99.75% w/ a normal logistic regresssion model
        # on this dataset, get 98.83% with this model

    def test_binaryclassification_regularized(self):
        # Sanity check - high regularization leads to very high losses.
        multi_layer_perceptron1 = MultiLayerPerceptron(
            typeSupervised="binary",
            numberInputFeatures=self.x_train.shape[0],
            regularization="L1",
            reg_parameter=500)
        multi_layer_perceptron2 = MultiLayerPerceptron(
            typeSupervised="binary",
            numberInputFeatures=self.x_train.shape[0],
            regularization="L2",
            reg_parameter=500)
        # 10 features learnt in first hidden layer w/ ReLU
        multi_layer_perceptron1.add_layer(10, activation_function=ReLU())
        # # 5 features learnt in second hidden layer w/ ReLU
        multi_layer_perceptron1.add_layer(5, activation_function=ReLU())
        # # Output layer sigmoid activation
        multi_layer_perceptron1.add_layer(1, activation_function=Sigmoid())

        # 10 features learnt in first hidden layer w/ ReLU
        multi_layer_perceptron2.add_layer(10, activation_function=ReLU())
        # # 5 features learnt in second hidden layer w/ ReLU
        multi_layer_perceptron2.add_layer(5, activation_function=ReLU())
        # # Output layer sigmoid activation
        multi_layer_perceptron2.add_layer(1, activation_function=Sigmoid())

        train_loss1, valid_loss1, train_acc1, valid_acc1 = (
            multi_layer_perceptron1.fit(self.x_train,
                                        self.y_train,
                                        self.x_valid,
                                        self.y_valid,
                                        ret_train_loss=True,
                                        num_epochs=10,
                                        learn_rate=2.6))
        preds1 = multi_layer_perceptron1.predict_multi_layer_perceptron(
            self.x_test, 0.5)
        acc1 = accuracy(self.y_test, preds1)
        print(acc1)
        print(train_loss1)
        print("\n")
        print(valid_loss1)

        train_loss2, valid_loss2, train_acc2, valid_acc2 = (
            multi_layer_perceptron2.fit(self.x_train,
                                        self.y_train,
                                        self.x_valid,
                                        self.y_valid,
                                        ret_train_loss=True,
                                        num_epochs=10,
                                        learn_rate=3))
        preds2 = multi_layer_perceptron2.predict_multi_layer_perceptron(
            self.x_test, 0.5)
        acc2 = accuracy(self.y_test, preds2)
        print(acc2)
        print(train_loss2)
        print("\n")
        print(valid_loss2)

    def testNormalRegBinary(self):
        """ You have to be really careful with the reg parameter. If its to high
        (past 0.1), your network won't learn anything. With softmax regression
        and logistic regression, there existed leeway to make the reg parameter
        pretty high but that absolutely won"t work with neural networks multiple
        layers deep.
        """
        # normal regularization
        MLP3 = MultiLayerPerceptron(typeSupervised="binary",
                                    numberInputFeatures=self.x_train.shape[0],
                                    regularization="L1",
                                    reg_parameter=0.01)
        MLP4 = MultiLayerPerceptron(typeSupervised="binary",
                                    numberInputFeatures=self.x_train.shape[0],
                                    regularization="L2",
                                    reg_parameter=0.01)

        # 10 features learnt in first hidden layer w/ ReLU
        MLP3.add_layer(10, activation_function=ReLU())
        # # 5 features learnt in second hidden layer w/ ReLU
        MLP3.add_layer(5, activation_function=ReLU())
        # # Output layer sigmoid activation
        MLP3.add_layer(1, activation_function=Sigmoid())

        # 10 features learnt in first hidden layer w/ ReLU
        MLP4.add_layer(10, activation_function=ReLU())
        # # 5 features learnt in second hidden layer w/ ReLU
        MLP4.add_layer(5, activation_function=ReLU())
        # # Output layer sigmoid activation
        MLP4.add_layer(1, activation_function=Sigmoid())

        train_loss3, valid_loss3, train_acc3, valid_acc3 = MLP3.fit(
            self.x_train,
            self.y_train,
            self.x_valid,
            self.y_valid,
            ret_train_loss=True,
            num_epochs=100,
            learn_rate=2.8)
        preds3 = MLP3.predict_multi_layer_perceptron(self.x_test, 0.5)
        acc3 = accuracy(self.y_test, preds3)
        print(acc3)
        print("\n")
        print(train_loss3)
        print("\n")
        print(valid_loss3)
        print("\n")

        train_loss4, valid_loss4, train_acc4, valid_acc4 = MLP4.fit(
            self.x_train,
            self.y_train,
            self.x_valid,
            self.y_valid,
            ret_train_loss=True,
            num_epochs=100,
            learn_rate=2.6)
        preds4 = MLP4.predict_multi_layer_perceptron(self.x_test, 0.5)
        acc4 = accuracy(self.y_test, preds4)
        print(acc4)
        print(train_loss4)
        print("\n")
        print(valid_loss4)
        print("\n")

    def testMultiClass(self):
        # replicating results from CS231N on this dataset
        x, y, notEncodedy = create_spiral_dataset()
        sm = SoftmaxRegression(inLayerNeuron=2, numClasses=3)
        train_loss_sm, train_acc_sm = sm.fit(x,
                                             y,
                                             ret_train_loss=True,
                                             num_epochs=190,
                                             learn_rate=1)
        print(train_loss_sm)
        preds_sm = sm.predict(x)
        acc_sm = accuracy(notEncodedy, preds_sm)
        print(acc_sm)
        # Loss obtained from CS231N - ours should hopefully be better!
        self.assertLessEqual(train_loss_sm[-1], 0.786302)
        self.assertGreaterEqual(acc_sm, 0.49)

        MLP6 = MultiLayerPerceptron(typeSupervised="multiclass",
                                    numberInputFeatures=2)
        # Learn 100 features in first hidden layer
        MLP6.add_layer(100, activation_function=ReLU())
        # Output layer learn 3 features for softmax
        MLP6.add_layer(3, activation_function=Softmax(), isSoftmax=1)
        train_loss6, train_acc = MLP6.fit(xtrain=x,
                                          ytrain=y,
                                          num_epochs=1000,
                                          learn_rate=1,
                                          ret_train_loss=True)
        print(train_loss6)
        preds_MLP6 = MLP6.predict_multi_layer_perceptron(x)
        acc_6 = accuracy(notEncodedy, preds_MLP6)
        print(acc_6)
        # Performance without regularization should be above 90%
        self.assertLessEqual(train_loss6[-1], 0.245)
        self.assertGreaterEqual(acc_6, 0.90)

        # Performance with L2 regularization should be much better
        MLP7 = MultiLayerPerceptron(typeSupervised="multiclass",
                                    numberInputFeatures=2,
                                    regularization="L2",
                                    reg_parameter=1e-3)
        # Learn 100 features in first hidden layer
        MLP7.add_layer(100, activation_function=ReLU())
        # Output layer learn 3 features for softmax
        MLP7.add_layer(3, activation_function=Softmax(), isSoftmax=1)
        train_loss7, train_acc7 = MLP7.fit(xtrain=x,
                                           ytrain=y,
                                           num_epochs=5000,
                                           learn_rate=1,
                                           ret_train_loss=True)
        print(train_loss7, train_acc7)
        preds_MLP7 = MLP7.predict_multi_layer_perceptron(x)
        acc_7 = accuracy(notEncodedy, preds_MLP7)
        print(acc_7)

        # Performance with regularization should be approx 99%
        self.assertLessEqual(train_loss7[-1], 0.40)
        self.assertGreaterEqual(acc_7, 0.98)

        # Checked the implemmentation of our MLP against the reference
        # implementation on CS231N and looks fine!


if __name__ == "__main__":
    unittest.main()
