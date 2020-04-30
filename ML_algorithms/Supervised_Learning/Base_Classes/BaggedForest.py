import numpy as np 
import sys
sys.path.append("/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Supervised_Learning/Classifiers")
sys.path.append("/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Utility")
sys.path.append("/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Supervised_Learning/Regression")
from RegressionTree import RegressionTree
from ClassificationTree import ClassificationTree
from DecisionTreeFunctions import predictionClassification
from DecisionTreeFunctions import predictionRegression
from ScoreFunctions import MSE
from ScoreFunctions import RMSE
from ScoreFunctions import accuracy

class BaggedForest(object):
    """
    This class represents bootstrap aggregated (bagged) decision trees.

    Parameters:
    -> typeSupervised (0 or 1): If 0, train classification models. If 1, train regression models. 
    
    -> criterion (str): If classification, should be "gini" or "entropy. If regression, can be None. 
    
    -> num_estimators (int): The number of estimators to use in the ensemble

    -> max_samples (None, int, or float): The number of samples to put into each bootstrap sample. If None, the num samples
    is equal to the dataset size. If int, the number of samples is equal to the size of the int. If float, the float should be between
    0 and 1 and it will indicate the percentage to include in each bootstrap sample. 

    -> bootstrap (boolean): Boolean indicating whether to bootstrap the dataset when building the ensemble.

    -> minSamplesSplit (int): Integer indicating the minimum number of examples that have to fall in this node to justify splitting further
    
    -> maxDepth (int): Integer representing the maximum depth to grow this tree 
    
    -> maxFeatures (int): Integer representing the maximum number of features to use to determine the split
    
    -> min_impurity_decrease (int): The minimum decrease in impurity to justify splitting the node 
    
    -> verbose (boolean): Whether to give updates when fitting and predicting 
    """
    def __init__(self, typeSupervised, criterion, num_estimators = 100, max_samples = None, bootstrap = False, minSamplesSplit = 2, maxDepth = None, maxFeatures = None, min_impurity_decrease =0, verbose = False):
        self.typeSupervised = typeSupervised
        self.criterion = criterion 
        self.num_estimators = num_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.minSamplesSplit = minSamplesSplit
        self.maxDepth = maxDepth 
        self.maxFeatures = maxFeatures
        self.min_impurity_decrease = min_impurity_decrease
        self.verbose = verbose
        # variable that keeps track of the examples used to train every tree in the ensemble 
        # so we can get OOB accuracy and error
        self.examplesUsedInBootstrap = [] 

        # variable indicating if this bagged forest is a random forest 
        # ensemble holder
        self.forest = self._buildForest()


    def _buildForest(self):
        """
        This method builds the forest of trees out using the specifications given in the constructor.

        The individual trees will be trained when the .fit() method is called. 
        """
        forest = []
        # If classification, fit classification trees with the appropriate criterion
        # otherwise, fit regression trees 
        if self.typeSupervised == 0:
            criterion = True if self.criterion == "entropy" else False 
            for i in range(self.num_estimators):
                forest.append(ClassificationTree(entropy= criterion, minSamplesSplit=self.minSamplesSplit, maxDepth=self.maxDepth, maxFeatures=self.maxFeatures, min_impurity_decrease=self.min_impurity_decrease))
            return forest
        else:
            for i in range(self.num_estimators):
                forest.append(RegressionTree(minSamplesSplit=self.minSamplesSplit, maxDepth=self.maxDepth, maxFeatures=self.maxFeatures, min_impurity_decrease=self.min_impurity_decrease))
            return forest 

    def fit(self, xtrain, ytrain):
        """
        This method implements the .fit() method for bagged forests, where we build the forest on xtrain and ytrain. 

        Parameters:
        -> xtrain (NumPy matrix): A (N,M) matrix where N is features and M is examples 
        -> ytrain (NumPy vector): A (1,M) vector where M is the number of examples 

        Returns: None 
        """
        # Determine the size of each sample with which to train every tree 
        if type(self.max_samples) is int:
            bootstrapSize = self.max_samples 
        elif type(self.max_samples) is float:
            bootstrapSize = int(self.max_samples * xtrain.shape[1])
        else:
            bootstrapSize = xtrain.shape[1] 

        for i in range(len(self.forest)):
            # If we are bootstrapping, then we need to generate a bootstrap sample, record the examples used to train the
            # tree, and fit the tree on the boostrapped samples with the boostrapSize examples

            # If we aren't boostrapping, just fit every tree on the entire training set 
            if self.bootstrap:
                # sample with replacement when we decide which examples to use to train the ith tree
                examplesInBootstrappedSample = np.random.choice(xtrain.shape[1], size = bootstrapSize, replace = True)
                xBootstrapped = xtrain[:, examplesInBootstrappedSample]
                yBootStrapped = ytrain[:, examplesInBootstrappedSample]
                self.examplesUsedInBootstrap.append(set(examplesInBootstrappedSample))
                self.forest[i].fit(xBootstrapped, yBootStrapped)
            else:
                self.forest[i].fit(xtrain, ytrain)
            
            if self.verbose:
                print("Finished fitting the %sst tree in the ensemble, %s/%s of the way there!"%(i, i, len(self.forest)))
                print('\n')
        
    def predict(self, x):
        """
        This method implements the .predict() method for bagged forests. In the .predict() method, we simply
        aggregate the predictions from each individual tree in the forest. 

        Parameters:
        -> x (NumPy matrix): A (N,M) matrix where N is features and M is examples 

        Returns: None 
        """

        predictions = np.zeros((1, x.shape[1]))
        for i in range(x.shape[1]):
            feature_vector = x[:,i].reshape(-1,1)
            feature_vectorPreds = np.zeros((1, x.shape[1]))
            for j in range(len(self.forest)):
                predictionIthTree = self.forest[i].predict(feature_vector)
                feature_vectorPreds[:,i] = predictionIthTree
            if self.typeSupervised == 0:
                singlePrediction = predictionClassification(feature_vectorPreds)
            else:
                singlePrediction = predictionRegression(feature_vectorPreds)
            predictions[:,i] = singlePrediction
            if self.verbose:
                print("Finished predicting on the %sst example, %s/%s of the way there!"%(i, i, len(self.forest)))
                print('\n')
        return predictions

    def getOOBScore(self, xtrain, ytrain):
        """
        This method gets the out of bag score for the model. The method loops over every single example 
        in the training set, and for every example, if the current tree did NOT fit on it, then the tree will
        predict on it. The predictions for every tree will be amalgamated into one prediction. 

        For classification, we will return the accuracy and the error on the out of bag sample.
        For regression, we will return the mean squared error(mse) and the root mean squared error (rmse)
        on the out of bag sample.

        Parameters:
        -> xtrain (NumPy matrix): A (N,M) matrix where N is features and M is examples 
        -> ytrain (NumPy vector): A (1,M) vector where M is the number of examples 

        Returns:
        -> If self.typeSupervised == 0, accuracy (int) and error (int)
        -> If self.typeSupervised == 1, mse (int) and rmse (int)
        """
        predictions = np.zeros((1, xtrain.shape[1]))
        for i in range(xtrain.shape[1]):
            feature_vector = xtrain[:,i].reshape(-1, 1)
            preds_currVector = []  
            for j in range(len(self.forest)):
                # if this tree trained on this example, then skip it 
                if i in self.examplesUsedInBootstrap[j]:
                    continue
                # otherwise predict on it 
                prediction = self.forest[j].predict(feature_vector)
                preds_currVector.append(prediction)
            if self.typeSupervised == 0:
                overallPrediction = predictionClassification(preds_currVector)
            else:
                overallPrediction = predictionRegression(preds_currVector)
            predictions[:,i] = overallPrediction
        if self.typeSupervised == 0:
            acc = accuracy(ytrain, predictions)
            error = 1-acc
            return acc, error
        else:
            mse = MSE(ytrain, predictions)
            rmse = RMSE(ytrain, predictions)
            return mse, rmse 







            



        
        
        
    

