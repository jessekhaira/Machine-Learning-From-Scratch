import numpy as np
from  ML_algorithms.Neural_Net_Util.NeuralNetwork_Base import NeuralNetwork_Base
from  ML_algorithms.Neural_Net_Util.LossFunctions import negative_log_loss
from ML_algorithms.Neural_Net_Util.ActivationFunctions import Sigmoid
import copy 

class LogisticRegression(NeuralNetwork_Base):
    """
    This class is a template to create logistic regresssion objects. 
    The logistic regression object is used for binary classification tasks, where
    the output labels are integers. The object contains one hidden layer with one 
    neuron using the sigmoid activation function. 

    Parameters:
    -> inLayerNeuron (int): Integer representing how many features are at the input to the classifier
    -> classificationThreshold (int): Scalar value that is applied to the predictions, to separate the positive class
    from the negative class when predicting. 
    -> regularization (str): Type of regularization to use. Either "L2" or "L1" is accepted.
    -> regParameter(int): Integer representing the strength of the regularization
    """
    def __init__(self, inLayerNeuron, classificationThreshold = None, regularization = None, regParameter = None):
        loss_obj = negative_log_loss(regularization, regParameter)
        activ = Sigmoid() 
        super(LogisticRegression, self).__init__(input_features = inLayerNeuron, lossFunction = loss_obj)
        # Logistic regression has one fully connected layer, with a single neuron, with the sigmoid
        # activation function 
        self.add_layer(1, activ)     
        self.classificationThreshold = classificationThreshold

    def predict(self, X):
        predictions = self._forward_propagate(X)
        if self.classificationThreshold:
            return predictions >= self.classificationThreshold
        return predictions

    
class OneVsAllLogisticRegression(object):
    """
    This class is a template to create OneVsAll (OVA) logistic regresssion objects. 
    The OVA regression object is used for multi class classification tasks, where
    the output labels are integers. The object contains N logistic regression objects,
    each trained to identify one type of class. 

    Parameters:
    -> num_classes (int): The number of classes in your dataset
    -> num_in_neurons (int): The number of features in your dataset
    -> num_epochs (int): The number of epochs you would like to train your N objects for 
    -> learn_rate (int): The speed at which to update parameters during gradient descent 
    """
    def __init__(self, num_classes, num_in_neurons, num_epochs, learn_rate):
        self.model = []
        for i in range(num_classes):
            self.model.append(LogisticRegression(num_in_neurons))
        self.datasets = []
        self.num_epochs = num_epochs
        self.learn_rate = learn_rate

    def fit(self, xtrain, ytrain):
        self._buildDatasets(xtrain, ytrain)
        for i in range(len(self.model)):
            self.model[i].fit(self.datasets[i][0], self.datasets[i][1], num_epochs= self.num_epochs, learn_rate=self.learn_rate)

    def _buildDatasets(self, xtrain, ytrain):
        classes_data = np.unique(ytrain)
        for i in range(len(classes_data)):
            curr_class = classes_data[i]
            only_one_labelis1 = (ytrain == curr_class).astype(int)
            self.datasets.append((xtrain, only_one_labelis1))


    def predict(self, x):
        assert x.shape[0] == self.model[0].num_input, "Your new data has to have as many features as what you trained on"
        predictions = [] 
        for i in range(len(self.model)):
            predictions.append(self.model[i].predict(x))
        # Stack all predictions next to each other in a matrix so we can easily get col vals 
        # for each row using np.argmax()
        # Predictions from each unit will be a (1,M) vector, so we need to stack them up in rows
        # and then get max row val for each example (Class) by saying np.argmax(axis=0)
        matrix_pred = np.row_stack((i for i in predictions))
        # matrix_pred should be of shape (num_features in example, num examples * num predictors)
        assert matrix_pred.shape == (len(self.model), x.shape[1])
        final_output = np.argmax(matrix_pred, axis = 0)
        assert final_output.shape == (x.shape[1],)
        return final_output