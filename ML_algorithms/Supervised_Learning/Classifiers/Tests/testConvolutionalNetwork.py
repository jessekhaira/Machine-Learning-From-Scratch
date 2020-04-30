import sys
sys.path.append("/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Supervised_Learning/Classifiers")
sys.path.append("/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Neural_Net_Util")
sys.path.append("/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Utility")
from MultiLayerPerceptron import MultiLayerPerceptron
from NeuralNet_Layers import DenseLayer
from SoftmaxRegression import softmax_regression
from ConvolutionalLayers import Conv2D 
from ConvolutionalLayers import Pool
from ConvolutionalNeuralNet import ConvolutionalNeuralNetwork
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


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.array(x_train, dtype = float)
x_test = np.array(x_test, dtype = float)
x_train /= 255
x_test /= 255 
x_train = x_train
x_test = x_test
saved_y = y_train.reshape(1,-1)
y_train = oneHotEncode(y_train.reshape(1,-1))
y_test = oneHotEncode(y_test.reshape(1,-1))


x_miniTrain = x_train[:1000,:,:].reshape(1000,1,28,28)
y_miniTrain = y_train[:,:1000]
x_miniValid = x_train[:1000,:,:].reshape(1000,1,28,28)
y_miniValid =  y_train[:,1000:2000]

print(type(Conv2D))

obj1 = ConvolutionalNeuralNetwork(typeSupervised="multiclass", inputDepth=1)

paramsLayer1 = {"filterSize":3, "inputDepth":None, "numFilters":5, "activationFunction": ReLU(), "padding":"same", "stride":1, "finalConvLayer":False}
obj1.addConvNetLayer(Conv2D, **paramsLayer1)
paramsLayer2 = {"filterSize":3, "stride":2,"finalConvLayer":False, "poolType":"avg", "padding":"valid"}
obj1.addConvNetLayer(Pool, **paramsLayer2)


paramsLayer3 = {"filterSize":3, "inputDepth":None, "numFilters":5, "activationFunction": ReLU(), "padding":"same", "stride":1, "finalConvLayer":False}
obj1.addConvNetLayer(Conv2D, **paramsLayer3)
paramsLayer4 = {"filterSize":3, "finalConvLayer":True, "stride":2,"finalConvLayer":True,"poolType":"avg", "padding":"valid"}
obj1.addConvNetLayer(Pool, **paramsLayer4)

paramsLayer5 = {"num_neurons":75, "activationFunction":ReLU(), "regularization":None, "regParameter":None}
obj1.addConvNetLayer(DenseLayer, **paramsLayer5)

paramsLayer6 = {"num_neurons":10, "activationFunction":Softmax(), "regularization":None, "regParameter":None, "isSoftmax":1}
obj1.addConvNetLayer(DenseLayer, **paramsLayer6)

class testConvNet_Mnist(unittest.TestCase):
    def testForwardPass(self):
        # just making sure that everything connects properly and 
        # making sure that predictions are all 0.1 for every example 
        preds = obj1._forward_propagate(x_miniTrain[:32])
        print(preds)

    def testOverfitSmallBatch_smallNet(self):
        # We're gonna test if we can overfit the above conv net on a small batch w/ just one layer followed by a softmax classifier
        # if we can't.. then somethings wrong with the backprop for the conv layer 
        obj2 = ConvolutionalNeuralNetwork(typeSupervised="multiclass", inputDepth=1)
        paramsLayer1 = {"filterSize":3, "inputDepth":None, "numFilters":5, "activationFunction": ReLU(), "padding":"same", "stride":1, "finalConvLayer":True}
        obj2.addConvNetLayer(Conv2D, **paramsLayer1)


        paramsLayer6 = {"num_neurons":10, "activationFunction":Softmax(), "regularization":None, "regParameter":None, "isSoftmax":1}
        obj2.addConvNetLayer(DenseLayer, **paramsLayer6)

        train_loss, train_acc = obj2.fit(x_miniTrain[:32], y_miniTrain[:,:32], num_epochs=350, ret_train_loss= True, verbose=True, learn_rate=0.4, optim=gradientDescentMomentum())
        print(train_loss)
        
        # Looks good! 

    def testOverfitSmallBatch_medNetAvgPool(self):
        # testing conv layer followed by avg pool layer followed by classifier
        obj3 = ConvolutionalNeuralNetwork(typeSupervised="multiclass", inputDepth=1)
        paramsLayer1 = {"filterSize":3, "inputDepth":None, "numFilters":5, "activationFunction": ReLU(), "padding":"same", "stride":1, "finalConvLayer":False}
        obj3.addConvNetLayer(Conv2D, **paramsLayer1)

        paramsLayer2 = {"filterSize":3, "stride":2,"finalConvLayer":True, "poolType":"avg", "padding":"valid"}
        obj3.addConvNetLayer(Pool, **paramsLayer2)

        paramsLayer6 = {"num_neurons":10, "activationFunction":Softmax(), "regularization":None, "regParameter":None, "isSoftmax":1}
        obj3.addConvNetLayer(DenseLayer, **paramsLayer6)

        train_loss, train_acc = obj3.fit(x_miniTrain[:32], y_miniTrain[:,:32], num_epochs=500, ret_train_loss= True, verbose=True, learn_rate=0.4, optim=gradientDescentMomentum())
        
        # Looks good! 

    def testOverfitSmallBatch_medNetMaxPool(self):
        # testing conv layer followed by avg pool layer followed by classifier
        obj3 = ConvolutionalNeuralNetwork(typeSupervised="multiclass", inputDepth=1)
        paramsLayer1 = {"filterSize":3, "inputDepth":None, "numFilters":5, "activationFunction": ReLU(), "padding":"same", "stride":1, "finalConvLayer":False}
        obj3.addConvNetLayer(Conv2D, **paramsLayer1)

        paramsLayer2 = {"filterSize":3, "stride":2,"finalConvLayer":True, "poolType":"max", "padding":"valid"}
        obj3.addConvNetLayer(Pool, **paramsLayer2)

        paramsLayer6 = {"num_neurons":10, "activationFunction":Softmax(), "regularization":None, "regParameter":None, "isSoftmax":1}
        obj3.addConvNetLayer(DenseLayer, **paramsLayer6)

        train_loss, train_acc = obj3.fit(x_miniTrain[:32], y_miniTrain[:,:32], num_epochs=500, ret_train_loss= True, verbose=True, learn_rate=0.4, optim=gradientDescentMomentum())
        
        # trains fine but loss goes down way slower than average pooling since we only send the gradient through the maximum neuron
        # versus every single neuron. Doesn't make a huge difference, just need to train for way longer. 

    def testFullNet(self):
        # train the full net - achieves great performance when allowed to train for a large number of epochs with a small learning rate
        # downside - takes an exceptionally long time to finish training, which is why its worth it to verify that conv layer + softmax layer and then
        # a conv layer + avg pool/max pool + softmax classifier can overfit to a small batch first which ensures that the forward prop and back prop equations are
        # wired properly. Deeper nets are then just stacks of these layers one after the other like obj1 is. 
        train_loss, train_acc = obj1.fit(x_train, y_train, num_epochs=40000, ret_train_loss= True, verbose=True, learn_rate=0.1, batch_size=128, optim=gradientDescentMomentum())

 

if __name__ == "__main__":
    unittest.main()

