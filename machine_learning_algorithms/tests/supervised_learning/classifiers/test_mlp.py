""" This module contains unit tests for the multi layer perceptron
algorithm """
import unittest
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from machine_learning_algorithms.utility.ScoreFunctions import accuracy
from machine_learning_algorithms.utility.misc import one_hot_encode
from machine_learning_algorithms.supervised_learning.classifiers.multi_layer_perceptron import MultiLayerPerceptron
from machine_learning_algorithms.supervised_learning.classifiers.softmax_regression import SoftmaxRegression
from machine_learning_algorithms.neural_net_utility.activation_functions import ReLU, Sigmoid, Softmax


def create_spiral_dataset():
    """ Function creates a 2 dimensional spiral dataset, returning
    100 feature vectors and encoded and unencoded labels for those
    vectors.
    """
    num_points = 100    # number of points per class
    dimensions = 2    # dimensionality
    num_classes = 3    # number of classes
    x = np.zeros((num_points * num_classes,
                  dimensions))    # data matrix (each row = single example)
    y = np.zeros(num_points * num_classes, dtype="uint8")    # class labels
    for j in range(num_classes):
        ix = range(num_points * j, num_points * (j + 1))
        r = np.linspace(0.0, 1, num_points)    # radius
        t = np.linspace(
            j * 4, (j + 1) * 4,
            num_points) + np.random.randn(num_points) * 0.2    # theta
        x[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    x = x.T
    not_encoded_y = y
    y = one_hot_encode(y.reshape(1, -1))
    return x, y, not_encoded_y


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
        MLP.fit(self.x_train,
                self.y_train,
                self.x_valid,
                self.y_valid,
                ret_train_loss=True,
                num_epochs=100,
                learn_rate=2.6)
        preds = MLP.predict_multi_layer_perceptron(self.x_test, 0.5)
        acc = accuracy(self.y_test, preds)
        self.assertGreaterEqual(acc, 0.95)

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

        multi_layer_perceptron1.fit(self.x_train,
                                    self.y_train,
                                    self.x_valid,
                                    self.y_valid,
                                    ret_train_loss=True,
                                    num_epochs=10,
                                    learn_rate=2.6)
        preds1 = multi_layer_perceptron1.predict_multi_layer_perceptron(
            self.x_test, 0.5)

        multi_layer_perceptron2.fit(self.x_train,
                                    self.y_train,
                                    self.x_valid,
                                    self.y_valid,
                                    ret_train_loss=True,
                                    num_epochs=10,
                                    learn_rate=3)
        preds2 = multi_layer_perceptron2.predict_multi_layer_perceptron(
            self.x_test, 0.5)

        acc1 = accuracy(self.y_test, preds1)
        acc2 = accuracy(self.y_test, preds2)
        self.assertLessEqual(acc1, 0.69)
        self.assertLessEqual(acc2, 0.69)

    def test_binaryclassifcation_regularized2(self):
        """ You have to be really careful with the reg parameter. If its to high
        (past 0.1), your network won't learn anything. With softmax regression
        and logistic regression, there existed leeway to make the reg parameter
        pretty high but that absolutely won"t work with neural networks multiple
        layers deep.
        """
        # normal regularization
        multi_layer_perceptron3 = MultiLayerPerceptron(
            typeSupervised="binary",
            numberInputFeatures=self.x_train.shape[0],
            regularization="L1",
            reg_parameter=0.01)
        multi_layer_perceptron4 = MultiLayerPerceptron(
            typeSupervised="binary",
            numberInputFeatures=self.x_train.shape[0],
            regularization="L2",
            reg_parameter=0.01)

        # 10 features learnt in first hidden layer w/ ReLU
        multi_layer_perceptron3.add_layer(10, activation_function=ReLU())
        # # 5 features learnt in second hidden layer w/ ReLU
        multi_layer_perceptron3.add_layer(5, activation_function=ReLU())
        # # Output layer sigmoid activation
        multi_layer_perceptron3.add_layer(1, activation_function=Sigmoid())

        # 10 features learnt in first hidden layer w/ ReLU
        multi_layer_perceptron4.add_layer(10, activation_function=ReLU())
        # # 5 features learnt in second hidden layer w/ ReLU
        multi_layer_perceptron4.add_layer(5, activation_function=ReLU())
        # # Output layer sigmoid activation
        multi_layer_perceptron4.add_layer(1, activation_function=Sigmoid())

        multi_layer_perceptron3.fit(self.x_train,
                                    self.y_train,
                                    self.x_valid,
                                    self.y_valid,
                                    ret_train_loss=True,
                                    num_epochs=100,
                                    learn_rate=2.8)
        preds3 = multi_layer_perceptron3.predict_multi_layer_perceptron(
            self.x_test, 0.5)
        multi_layer_perceptron4.fit(self.x_train,
                                    self.y_train,
                                    self.x_valid,
                                    self.y_valid,
                                    ret_train_loss=True,
                                    num_epochs=100,
                                    learn_rate=2.6)
        preds4 = multi_layer_perceptron4.predict_multi_layer_perceptron(
            self.x_test, 0.5)

        acc3 = accuracy(self.y_test, preds3)
        acc4 = accuracy(self.y_test, preds4)
        self.assertGreaterEqual(acc3, 0.95)
        self.assertGreaterEqual(acc4, 0.95)

    def test_multiclass_1(self):
        x, y, not_encoded_y = create_spiral_dataset()
        sm = SoftmaxRegression(inLayerNeuron=2, numClasses=3)
        train_loss_sm, _ = sm.fit(x,
                                  y,
                                  ret_train_loss=True,
                                  num_epochs=190,
                                  learn_rate=1)
        preds_sm = sm.predict(x)
        acc_sm = accuracy(not_encoded_y, preds_sm)
        self.assertLessEqual(train_loss_sm[-1], 0.786302)
        self.assertGreaterEqual(acc_sm, 0.40)

    def test_multi_class2(self):
        x, y, not_encoded_y = create_spiral_dataset()
        multi_layer_perceptron = MultiLayerPerceptron(
            typeSupervised="multiclass", numberInputFeatures=2)
        multi_layer_perceptron.add_layer(100, activation_function=ReLU())
        multi_layer_perceptron.add_layer(3,
                                         activation_function=Softmax(),
                                         isSoftmax=1)
        train_loss6, _ = multi_layer_perceptron.fit(xtrain=x,
                                                    ytrain=y,
                                                    num_epochs=1000,
                                                    learn_rate=1,
                                                    ret_train_loss=True)
        preds = multi_layer_perceptron.predict_multi_layer_perceptron(x)
        acc_6 = accuracy(not_encoded_y, preds)
        # Performance without regularization should be above 90%
        self.assertLessEqual(train_loss6[-1], 0.245)
        self.assertGreaterEqual(acc_6, 0.90)

    def test_multi_class3(self):
        # Performance with L2 regularization should be much better
        x, y, not_encoded_y = create_spiral_dataset()
        multi_layer_perceptron = MultiLayerPerceptron(
            typeSupervised="multiclass",
            numberInputFeatures=2,
            regularization="L2",
            reg_parameter=1e-3)
        # Learn 100 features in first hidden layer
        multi_layer_perceptron.add_layer(100, activation_function=ReLU())
        # Output layer learn 3 features for softmax
        multi_layer_perceptron.add_layer(3,
                                         activation_function=Softmax(),
                                         isSoftmax=1)
        train_loss7, _ = multi_layer_perceptron.fit(xtrain=x,
                                                    ytrain=y,
                                                    num_epochs=5000,
                                                    learn_rate=1,
                                                    ret_train_loss=True)
        preds = multi_layer_perceptron.predict_multi_layer_perceptron(x)
        acc_7 = accuracy(not_encoded_y, preds)
        self.assertLessEqual(train_loss7[-1], 0.40)
        self.assertGreaterEqual(acc_7, 0.98)


if __name__ == "__main__":
    unittest.main()
