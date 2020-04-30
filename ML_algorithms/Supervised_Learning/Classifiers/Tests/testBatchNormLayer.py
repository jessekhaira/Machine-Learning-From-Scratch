import sys
sys.path.append("/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Supervised_Learning/Classifiers")
sys.path.append("/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Neural_Net_Util")
sys.path.append("/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Utility")
from MultiLayerPerceptron import MultiLayerPerceptron
from NeuralNet_Layers import BatchNormLayer_Dense
from SoftmaxRegression import softmax_regression
from ActivationFunctions import ReLU
from ActivationFunctions import TanH
from ActivationFunctions import Sigmoid
from ActivationFunctions import Softmax
from Optimizers import gradientDescentMomentum
from Optimizers import Adam
from Optimizers import RMSProp 
import unittest
import numpy as np 
from ScoreFunctions import accuracy
from misc import oneHotEncode
import tensorflow as tf 
import unittest

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


x_train = np.array(x_train, dtype = float)
x_test = np.array(x_test, dtype = float)
x_train /= 255
x_test /= 255 
x_train = x_train.reshape(784,-1)
x_test = x_test.reshape(784,-1)
saved_y = y_train.reshape(1,-1)
y_train = oneHotEncode(y_train.reshape(1,-1))
y_test = oneHotEncode(y_test.reshape(1,-1))
print(y_train.shape)
print(y_test.shape)
print(x_train[:,:5])


x_miniTrain = x_train[:,:1000].reshape(784,-1)
y_miniTrain = y_train[:,:1000]
x_miniValid = x_train[:,1000:2000].reshape(784,-1)
y_miniValid =  y_train[:,1000:2000]

class testMNIST_BatchNormLayers(unittest.TestCase):
    def test_softmaxRegressor(self):
        sm = softmax_regression(784, 10)
        train_loss, train_acc = sm.fit(x_miniTrain, y_miniTrain, num_epochs=500, ret_train_loss=True)
    
    def testOverFitSmallBatch(self):
        MLP = MultiLayerPerceptron(typeSupervised = "multiclass", numberInputFeatures=784)

        MLP.add_layer(num_neurons = 100, activationFunction = ReLU(), layer = BatchNormLayer_Dense)
        MLP.add_layer(num_neurons=10, activationFunction = Softmax(), isSoftmax= True)

        train_loss1, train_acc1 = MLP.fit(x_train[:,:100].reshape(784,-1), y_train[:,:100], num_epochs = 150, ret_train_loss= True, optim=RMSProp(), learn_rate=0.001)
        predictions1 = MLP.predictMLP(x_train[:,:100].reshape(784,-1))
        acc = accuracy(saved_y[:,:100].reshape(1,-1), predictions1)
        print(train_loss1)
        print(acc)
        self.assertLessEqual(train_loss1[-1], 0.09)
        self.assertEqual(acc,1)

        MLP2 = MultiLayerPerceptron(typeSupervised = "multiclass", numberInputFeatures=784)

        MLP2.add_layer(num_neurons = 100, activationFunction = ReLU(), layer = BatchNormLayer_Dense)
        MLP2.add_layer(num_neurons=10, activationFunction = Softmax(), isSoftmax= True)

        train_loss2, train_acc2 = MLP2.fit(x_train[:,:100].reshape(784,-1), y_train[:,:100], num_epochs = 150, ret_train_loss= True, optim=gradientDescentMomentum(), learn_rate=0.1)
        predictions2 = MLP2.predictMLP(x_train[:,:100].reshape(784,-1))
        acc2 = accuracy(saved_y[:,:100].reshape(1,-1), predictions2)
        print(train_loss2)
        print(acc2)
        self.assertLessEqual(train_loss2[-1], 0.09)
        self.assertEqual(acc2,1)

    def testBiggerBatch_BatchNorm(self):
        MLP3 = MultiLayerPerceptron(typeSupervised = "multiclass", numberInputFeatures=784, regularization="L2", regParameter=0.01)

        MLP3.add_layer(num_neurons = 100, activationFunction = ReLU(), layer = BatchNormLayer_Dense)
        MLP3.add_layer(num_neurons = 100, activationFunction = ReLU(), layer = BatchNormLayer_Dense)
        MLP3.add_layer(num_neurons=10, activationFunction = Softmax(), isSoftmax= True)


        train_loss, valid_loss, train_acc, valid_acc = MLP3.fit(x_miniTrain, y_miniTrain, xvalid = x_miniValid, yvalid = y_miniValid,  num_epochs = 500, ret_train_loss= True, optim=gradientDescentMomentum(), learn_rate=0.1)
        print(train_loss)
        print('\n')
        print(valid_loss)
        print('\n')
        print(train_acc)
        print('\n')
        print(valid_acc)


    def testBiggerBatch_normal(self):
        MLP4 = MultiLayerPerceptron(typeSupervised = "multiclass", numberInputFeatures=784)

        MLP4.add_layer(num_neurons = 100, activationFunction = ReLU())
        MLP4.add_layer(num_neurons = 100, activationFunction = ReLU())
        MLP4.add_layer(num_neurons=10, activationFunction = Softmax(), isSoftmax= True)
        
        train_loss, valid_loss, train_acc, valid_acc = MLP4.fit(x_miniTrain, y_miniTrain, xvalid = x_miniValid, yvalid = y_miniValid,  num_epochs = 100, ret_train_loss= True, optim=gradientDescentMomentum(), learn_rate=0.1)
        print(train_loss)
        print('\n')
        print(valid_loss)
        print('\n')
        print(train_acc)
        print('\n')
        print(valid_acc)


## MLP with 2 batch norm layers with 100 neurons and a 10 way softmax each is fitting well to the training set but just overfitting
# -> fixed with some regulariization and exposing the model to more examples

## MLP with 2 normal layers with 100 neurons each and a 10 way softmax is not fitting well to the training set and not generalizing
# when trained with the same hyperparameters as the other network

# Effectiveness of batch norm layers! 
if __name__ == "__main__":
    unittest.main()