import numpy as np
from ML_algorithms.Neural_Net_Util.NeuralNet_Layers import DenseLayer
from ML_algorithms.Neural_Net_Util.NeuralNet_Layers import BatchNormLayer_Dense
from ML_algorithms.Neural_Net_Util.ConvolutionalLayers import Conv2D
from ML_algorithms.Neural_Net_Util.ConvolutionalLayers import Pool 
from ML_algorithms.Neural_Net_Util.Optimizers import gradientDescent
import random 
from ML_algorithms.Utility.misc import convertToHighestPred
import copy 
from ML_algorithms.Utility.ScoreFunctions import accuracy

    
class NeuralNetwork_Base(object):
    """
    This is a fully-connected Neural Network class, which can be used
    for supervised learning and unsupervised learning (autoencoders).

    Parameters:
    -> lossFunction (object) -> The lossFunction you want to use to train your network
    -> input_features (int) -> The number of input features a single example has in your 
    dataset. Fully connected nets are quite inflexible about this specification. 
    -> layers (python list) -> List object containing all the layers present in the neural
    network
    """
    def __init__(self, lossFunction, input_features):
        self.layers = [] 
        self.lossFunction = lossFunction 
        self.num_input = input_features
        
    def add_layer(self, num_neurons, activationFunction, isSoftmax = 0, layer = None, keep_prob=1):
        """
        This method adds a dense layer to your neural network. 

        Parameters:
        -> num_neurons (int) -> int representing the number of neurons you would like in this dense layer of the neural network. 
        -> activationFunction (object) -> Object which will be used to introduce non-linearity into your neural net
        -> isSoftmax (0 or 1): 0 or 1 indicating whether the activation function is softmax or not
        -> layer (object): type of layer being added to the neural network. Default layer is a Dense layer.

        Returns: None
        """
        if not self.layers:
            layer_x = DenseLayer(self.num_input, num_neurons, activationFunction, self.lossFunction.regularization, self.lossFunction.regParameter, isSoftmax) if layer is None else layer(self.num_input, num_neurons, activationFunction, self.lossFunction.regularization, self.lossFunction.regParameter, isSoftmax, keep_prob)
            self.layers.append(layer_x)
        else:
            # if the layeer beforee this is a dense layer, then get its weight shape
            # otherwise if its a conv layer/ pool layer, we have no idea how many neurons are going to be passed
            # to this layer so set it to None 
            shape_1 = self.layers[-1].W.shape[0] if (isinstance(self.layers[-1], DenseLayer) and self.layers[-1].W is not None) else None 
            layer_x = DenseLayer(shape_1, num_neurons, activationFunction, self.lossFunction.regularization, self.lossFunction.regParameter, isSoftmax) if layer is None else layer(shape_1, num_neurons, activationFunction, self.lossFunction.regularization, self.lossFunction.regParameter, isSoftmax, keep_prob)
            self.layers.append(layer_x)



    def fit(self, xtrain, ytrain, xvalid = None, yvalid = None, num_epochs =10, batch_size = 32, ret_train_loss = False, learn_rate = 0.1, optim = gradientDescent(), verbose = False):
        """
        This method trains the neural network on the training set.

        M: number of training examples
        N: number of features in a single example

        Parameters:
        -> xtrain (NumPy Matrix): Feature vectors of shape (M, N) 

        -> ytrain (NumPy Vector): Labels for the feature vectors of shape (M,1)

        -> xvalid (NumPy Matrix): Validation feature vectors of shape (M, N) 

        -> yvalid (NumPy Vector): Labels for the validation feature vectors of shape (M,1)

        -> num_epochs (int): Number of epochs to train the model

        -> batch_size (int): Number of examples to go through before performing a parameter update

        -> ret_train_loss (Boolean): Boolean value indicating whether to return train loss and valid loss
        if validation set provided 

        -> learn_rate (float): learning rate to be used when optimizing the loss function

        -> optim (function): optimizer to use to minimize the loss function 

        -> verbose (boolean): boolean value indicating whether to provide updates when training 

        Returns: None
        """
        # Dealing with edge case where you have less than 32 examples, which can happen maybe for k-fold cv
        # Just do batch gradient descent if the number of examples is super small 
        if xtrain.shape[1] <= 1000 and len(xtrain.shape)==2:
            batch_size = xtrain.shape[1]
            num_batches = 1 
        # otherwise do mini batch gradient descent
        elif xtrain.shape[1] <= 1000 and len(xtrain.shape)==2:
            num_batches = xtrain.shape[1]//batch_size
        else:
            num_batches = xtrain.shape[0]//batch_size
        train_loss = [] 
        train_acc = [] 
        validation_loss = [] 
        val_acc = [] 
        for epoch in range(num_epochs):
            currStart = 0
            currEnd = batch_size
            lossEpoch = [] 
            for i in range(num_batches):
                curr_y = ytrain[:,currStart:currEnd]
                if len(xtrain.shape) == 2:
                    curr_x = xtrain[:,currStart:currEnd]
                else:
                    # 3D pictures 
                    curr_x = xtrain[currStart:currEnd,:, :,:]                
                currStart = currEnd
                currEnd += batch_size 
                pred_miniBatch = self._forward_propagate(curr_x)
                loss = self._calculateLoss(curr_y, pred_miniBatch, self.layers)
                lossEpoch.append(loss)
                backpropInit = self.lossFunction.derivativeLoss_wrtPrediction(curr_y, pred_miniBatch)
                self._backward_propagate(backpropInit, learn_rate, optim, epoch, curr_x, curr_y)
            
            train_loss.append(np.mean(lossEpoch))

            if ytrain.shape[0] > 1:
                acc_trainSet = accuracy(convertToHighestPred(ytrain), self.predict(xtrain))
            else:
                acc_trainSet = accuracy(ytrain, self.predict(xtrain))
            train_acc.append(acc_trainSet)

            if xvalid is not None:
                if ytrain.shape[0] > 1:
                    acc_valSet = accuracy(convertToHighestPred(yvalid), self.predict(xvalid))
                    val_loss = self._calculateLoss(yvalid, self._forward_propagate(xvalid), self.layers)
                else:
                    acc_valSet = accuracy(yvalid, self.predict(xvalid))
                    val_loss = self._calculateLoss(yvalid, self.predict(xvalid), self.layers)
                
                val_acc.append(acc_valSet)
                validation_loss.append(val_loss)
            
            # provide updates during training for sanitys sake
            if verbose:
                print("Finished epoch %s"%(epoch))
                print("Train loss: %s, Train acc: %s"%(train_loss[-1], train_acc[-1]))
                if xvalid is not None:
                    print("Valid loss: %s, Valid acc: %s"%(validation_loss[-1], val_acc[-1]))
                

        if ret_train_loss and xvalid is not None:
            return train_loss, validation_loss, train_acc, val_acc 
        elif ret_train_loss:
            return train_loss, train_acc
    

    def predict(self, X, supervised = True):
        """
        This method is used to use the neural network to predict on instances it has not trained on.

        Parameters:
        -> X (NumPy matrix): NumPy matrix of shape (M,N) where M is the number of instances to predict
        on, and N is the number of features in an example.

        Returns: None
        """
        output = self._forward_propagate(X, train = False)
        # if more than one class, then compute the highest value as the prediction 
        if output.shape[0] > 1 and supervised:
            output = convertToHighestPred(output)
        return output 

    def _convertToHighestPred(self, predictions):
        predictions = np.argmax(predictions, axis=0)
        return predictions

    def _calculateLoss(self, curr_y, pred_minibatch, layersNet):
        """
        This method is used to calculate the loss of the neural network on a batch of
        examples that have been predicted on.

        M: Number of examples

        Parameters:
        -> curr_y (NumPy vector): NumPy vector of shape (M,1) consisting of the real answers for the 
        M examples
        -> pred_minibatch (NumPy vector): NumPy vector of shape (M,1) consisting of the predicted answers for 
        the M examples

        Returns: None
        """
        return self.lossFunction.get_loss(curr_y, pred_minibatch, layersNet)
    
    def _forward_propagate(self, X, train = True):
        """
        This method implements the forward propagation step for a neural network. 
        Each layer is fed in the activations from the previous layer a[L-1], a matrix that
        will be of shape (M, Nx) where M is the number of training examples and Nx is the 
        number of features, and computesits own activations a[L]. 
        These activations are fed to the next layer and so on.

        Parameters:
        - X (NumPy Matrix) -> A NumPy matrix of shape (M,Nx) where M is the number of outputs and
        Nx is the number of features.

        Returns: Output (NumPy Matrix) -> A NumPy matrix produced as output
        from the last hidden layer.
        """
        prev_activations = X
        for layer in self.layers:
            # if we are feeding in input from a Conv layer or pool layer
            # we don't know before how many activated neurons are going to be passed into
            # this dense layer, so we can't pre-initialize the weights for each layer.
            if isinstance(layer, DenseLayer) and layer.W is None:
                # conv layer will have flattened its output to matrix shape 
                layer.W, layer.b = layer._initializeWeights(layer.num_layer, prev_activations.shape[0])
                if isinstance(layer, BatchNormLayer_Dense):
                    layer.gamma, layer.beta = layer._initializeGammaBeta()
                    layer.runningMean, layer.runningVar = layer._initializeRunningMeanVar()
            elif (isinstance(layer, Conv2D)) and layer.filters is None:
                layer.inputDepth = prev_activations.shape[1]
                layer.filters, layer.b = layer._initializeWeights()
            activations = layer.compute_forward(prev_activations, train)
            prev_activations = activations
        return activations
    
    def _backward_propagate(self, initalGradient, learn_rate, optim, epoch, curr_x, curr_y):
        """
        This method implements the backward propagation step for a neural network. 
        The backpropagation is initialized by the gradient produced from the cost function
        dL/da. From there, we simply pass back through each of the layers in the neural network,
        with each layer computing dL/dZ, then from there getting dL/dW and dL/dB for this layer.
        The output from each layer will be dL/da[L-1], which is passed down further back in the circuit.

        Parameters:
        - initialGradient (NumPy matrix) -> This will represent the starting gradient dL/da.

        - learn_rate (float) -> learning rate to be used when optimizing the cost function
        Returns: None

        - optim (function) -> optimizer to use to minimize the loss function 
        """
        dLdA = initalGradient
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if not layer.optim:
                layer.optim = copy.deepcopy(optim)
            dLdA_prev = layer._updateWeights(dLdA, learn_rate, epoch, self, curr_x, curr_y, i)
            dLdA = dLdA_prev
        
        

        