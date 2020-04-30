import sys
sys.path.append("/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Supervised_Learning/Base_Classes")
from BaggedForest import BaggedForest

class RandomForestClassifier(BaggedForest):
    """
    This class represents a random forest regressor. 

    Parameters:       
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
    def __init__(self, criterion, num_estimators = 100, max_samples = None, bootstrap = False, minSamplesSplit = 2, maxDepth = None, maxFeatures = None, min_impurity_decrease =0, verbose = False):
        super(RandomForestClassifier, self).__init__(typeSupervised = 0, criterion = criterion, num_estimators=num_estimators, max_samples=max_samples, bootstrap=bootstrap, minSamplesSplit=minSamplesSplit, maxDepth=maxDepth, maxFeatures=maxFeatures, min_impurity_decrease=min_impurity_decrease, verbose=verbose)
