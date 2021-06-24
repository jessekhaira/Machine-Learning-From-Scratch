import numpy as np 
import random
from ML_algorithms.Utility.ScoreFunctions import MSE
from sklearn.metrics import log_loss

def regularization_loss(layersOfWeights, typeReg):
    reg_loss = 0 
    if typeReg == 'L2':
        for i in range(len(layersOfWeights)):
            reg_loss += (np.linalg.norm(layersOfWeights[i].W, ord = 2)**2)

    elif typeReg == 'L1':
        for i in range(len(layersOfWeights)):
            reg_loss += np.linalg.norm(layersOfWeights[i].W, ord =1)
    
    return reg_loss


class LossFunction(object):
    """
    This is the base LossFunction abstract class which all loss functions will inherit from.
    Every loss function will have a method of get_loss, derivativeLoss_wrtPrediction, and 
    _gradCheck, therefore it made sense to make an abstract class from which all these related
    classes will inherit from.
    """
    def get_loss(self, labels, predictions, layersOfWeights):
        raise NotImplementedError

    def derivativeLoss_wrtPrediction(self, labels, predictions):
        raise NotImplementedError

    def _gradCheck(self, labels, predictions, num_checks = 10):
        """
        This method does a quick gradient check to ensure the
        dL/dA is indeed correct. 

        Parameters:
        - labels (NumPy vector) -> (m,1) vector representing the labels for m examples

        - predictions (NumPy vector) -> (m,1) vector representing the predictions (prob between 0 and 1)
        for m examples
        
        - num_checks (int) -> number of times to check the gradient implentation

        Output (None)
        """
        eps = 1e-7
        random.seed(561)
        for i in range(num_checks):
            # Theoretically, we should be able to compute the loss with respect to every single
            # example at one time. It turns out that you lose precision when you do it that way 
            # so you don't get appropriate results for when you add eps and subtract eps AKA np.log
            # seems to not perform well in terms of precise accuracy when applied to an entire vector
            #  Thus, we just compute  our loss with a single example at a time as this seems to preserve accuracy much 
            # better and proves the gradient!  
            if labels.shape[0] >1:
                # Reshape multiclass labels to be easier to work with
                labels = labels.reshape(1, -1)
                predictions = predictions.reshape(1, -1)
            changeIdx = np.random.randint(0, labels.shape[1])
            y = labels[:, changeIdx].reshape(1,1)
            p = predictions[:, changeIdx].reshape(1,1)
            p_upeps = p+eps
            loss_higher = self.get_loss(y, p_upeps, None)
            p_downeps = p-eps
            loss_lower = self.get_loss(y, p_downeps, None)
            grad_analytic = self.derivativeLoss_wrtPrediction(y, p)
            grad_numeric = (loss_higher-loss_lower)/(2*eps)
            rel_error = abs(grad_analytic-grad_numeric)/abs(grad_analytic+grad_numeric+eps)
            print('rel error is %s'%(rel_error))

        
class negative_log_loss(LossFunction):
    """
    This class represents the negative log loss, which is typically the cost function to be optimized
    in binary classification tasks.

    Parameters:
    - regularization (string) -> Indicating which type of regularization you want to use, either "L2" or "L1"
    - regParameter (int) -> Integer indicating the strength of the regularization 
    """

    def __init__(self, regularization = None, regParameter = None):
        self.regularization = regularization
        self.regParameter = regParameter


    def get_loss(self, labels, predictions, layersOfWeights):      
        """
        Parameters:
        - labels (NumPy vector) -> (m,1) vector representing the labels for m examples

        - predictions (NumPy vector) -> (m,1) vector representing the predictions (prob between 0 and 1)
        for m examples
        """
        assert labels.shape == predictions.shape, "Somethings wrong, your labels have to be the same shape as the predictions!"
        # Numerical stability issues -> we never want to take the log of 0 so we clip our predictions at a lowest val of 1e-10
        predictions = np.clip(predictions,1e-10, 1-1e-10)
        data_loss = -(labels*np.log(predictions) + (1-labels)*np.log(1-predictions))
        # Cost is averaged overall all examples so we get
        # Tot_cost_batch = 1/m * (loss_examples_batch + reg_loss_batch)
        # Tot_cost_batch = (1/m) * loss_examples_batch + (1/m)*reg_loss_batch
        reg_loss = regularization_loss(layersOfWeights, self.regularization)
        if self.regularization == 'L2':
            return np.mean(data_loss + (self.regParameter/2)*reg_loss)

        # One examples loss, say zeroth, is -(y0*log(yhat0) + (1-y0)*log(1-yhat0) + lambda*(L1 norm or L2 norm))
        # The entire loss is this summed up over the entire vector of predictions
        # This operations has beeen vectorized to allow this to happen 
        elif self.regularization == 'L1':
            return np.mean(data_loss + self.regParameter*reg_loss)
        
        # no regularization, just return mean of data loss 
        return np.mean(data_loss)


    def derivativeLoss_wrtPrediction(self, labels, predictions):
        """
        This method represents the derivative of the cost function with respect to 
        the input y^ value. This gradient is meant to be passed back in the circuit
        of the neural network, and if there is regularization, the regularization will
        be included when updating the weights of a certain layer. 
        
        Parameters:
        labels (NumPy vector) -> (1,m) vector representing the labels for m examples

        predictions (NumPy vector) -> (1,m) vector representing the predictions (prob between 0 and 1)
        for m examples
        
        Output (NumPy vector) -> NumPy vector of shape (1,m) 
        """
        assert labels.shape == predictions.shape, "Somethings wrong, your labels have to be the same shape as the predictions!"   
        predictions = np.clip(predictions, 1e-10, 1-1e-10)
        # Include 1/batchsize term here for when we backprop 
        dLdA = 1/labels.shape[1]*((predictions-labels)/(predictions*(1-predictions)))
        return dLdA
    


        
class mean_squared_error(LossFunction):
    """
    This class represents the mean squared error loss, which is typically the cost function to be optimized
    in regression tasks.

    Parameters:
    - regularization (string) -> Indicating which type of regularization you want to use, either "L2" or "L1"
    - regParameter (int) -> Integer indicating the strength of the regularization 
    """
    def __init__(self, regularization = None, regParameter = None):
        self.regularization = regularization
        self.regParameter = regParameter 

    def get_loss(self, labels, predictions, layersOfWeights):
        """  
        Parameters:
        - labels (NumPy vector) -> (m,1) vector representing the labels for m examples

        - predictions (NumPy vector) -> (m,1) vector representing the predictions (prob between 0 and 1)
        for m examples
        """
        assert labels.shape == predictions.shape, "Somethings wrong, your labels have to be the same shape as the predictions!"
        # Numerical stability issues -> we never want to take the log of 0 so we clip our predictions at a lowest val of 1e-10
        data_loss = (1/2)*np.square(np.subtract(labels, predictions))
        # Cost is averaged overall all examples so we get
        # Tot_cost_batch = 1/m * (loss_examples_batch + reg_loss_batch)
        # Tot_cost_batch = (1/m) * loss_examples_batch + (1/m)*reg_loss_batch
        reg_loss = regularization_loss(layersOfWeights, self.regularization)
        if self.regularization == 'L2':
            return np.mean(data_loss + (self.regParameter/2)*reg_loss)

        # One examples loss, say zeroth, is -(y0*log(yhat0) + (1-y0)*log(1-yhat0) + lambda*(L1 norm or L2 norm))
        # The entire loss is this summed up over the entire vector of predictions
        # This operations has beeen vectorized to allow this to happen 
        elif self.regularization == 'L1':
            return np.mean(data_loss + self.regParameter*reg_loss)
        
        # no regularization, just return mean of data loss 
        return np.mean(data_loss)
    
    def derivativeLoss_wrtPrediction(self, labels, predictions):
        """
        This method represents the derivative of the cost function with respect to 
        the input y^ value. This gradient is meant to be passed back in the circuit
        of the neural network, and if there is regularization, the regularization will
        be included when updating the weights of a certain layer. 
        
        Parameters:
        labels (NumPy vector) -> (m,1) vector representing the labels for m examples

        predictions (NumPy vector) -> (m,1) vector representing the predictions (prob between 0 and 1)
        for m examples
        
        Output (NumPy vector) -> NumPy vector of shape (m,1) 
        """
        assert labels.shape == predictions.shape, "Somethings wrong, your labels have to be the same shape as the predictions!"   
        dLda = (1/labels.shape[1])*(predictions-labels)
        return dLda 
    

class cross_entropy(LossFunction):
    """
    This class represents the cross entropy loss, which is typically the cost function to be optimized
    in multiclass classification tasks.

    This cost function relies on the input being a (C,m) probability distribution as the same shape as
    the labels, where C is the number of classes you have in your data and m is the number of examples. 

    Parameters:
    - regularization (string) -> Indicating which type of regularization you want to use, either "L2" or "L1"
    - regParameter (int) -> Integer indicating the strength of the regularization 
    """
    def __init__(self, regularization = None, regParameter = None):
        self.regularization = regularization
        self.regParameter = regParameter 

    def get_loss(self, labels, predictions, layersOfWeights = None):
        """
        Parameters:
        - labels (NumPy matrix) -> (C,m) matrix representing the C labels for M examples

        - predictions (NumPy matrix) -> (C,m) matrix representing the softmax probability distribution 
        """
        # Numerical stability issues -> we never want to take the log of 0 so we clip our predictions at a lowest val of 1e-10
        predictions = np.clip(predictions, 1e-10, 1-1e-10)
        data_loss = -(labels*np.log(predictions))
        # Cost is averaged overall all examples so we get
        # Tot_cost_batch = 1/m * (loss_examples_batch + reg_loss_batch)
        # Tot_cost_batch = (1/m) * loss_examples_batch + (1/m)*reg_loss_batch
        reg_loss = regularization_loss(layersOfWeights, self.regularization)
        if self.regularization == 'L2':
            return np.mean(data_loss + (self.regParameter/2)*reg_loss)

        # One examples loss, say zeroth, is -(y0*log(yhat0) + (1-y0)*log(1-yhat0) + lambda*(L1 norm or L2 norm))
        # The entire loss is this summed up over the entire vector of predictions
        # This operations has beeen vectorized to allow this to happen 
        elif self.regularization == 'L1':
            return np.mean(data_loss + self.regParameter*reg_loss)
        
        # sum up all the losses for every single example (column wise sum) and then average them and return 
        return np.mean(np.sum(data_loss,axis=0))
    
    def derivativeLoss_wrtPrediction(self, labels, predictions):
        """
        This method represents the derivative of the cost function with respect to 
        the input y^ value. This gradient is meant to be passed back in the circuit
        of the neural network, and if there is regularization, the regularization will
        be included when updating the weights of a certain layer. 
        
        Parameters:
        - labels (NumPy matrix) -> (C,m) matrix representing the C labels for M examples

        - predictions (NumPy matrix) -> (C,m) matrix representing the softmax probability distribution 
        
        Output (NumPy matrix) -> NumPy matrix of shape (C,m) 
        """
        assert labels.shape == predictions.shape, "Somethings wrong, your labels have to be the same shape as the predictions!"   
        # -1/m dont forget in gradient! 
        dLda = -(1/labels.shape[1])*(labels/predictions)
        return dLda 




