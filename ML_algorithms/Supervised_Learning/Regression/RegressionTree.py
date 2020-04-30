import numpy as np 
import sys
sys.path.append("/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Supervised_Learning/Base_Classes")
sys.path.append("/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Utility")
from DecisionTree import BaseDecisionTree
from DecisionTreeFunctions import predictionRegression
from DecisionTreeFunctions import varianceReduction

class RegressionTree(BaseDecisionTree):
    """
    This class represents a regression decision tree trained using variance reduction. 

    Parameters:
    -> minSamplesSplit (int): Integer indicating the minimum number of examples that have to fall in this node to justify splitting further

    -> maxDepth (int): Integer representing the maximum depth to grow this tree 
    
    -> maxFeatures (int): Integer representing the maximum number of features to use to determine the split

    -> min_impurity_decrease (int): The minimum decrease in impurity to justify splitting a node 
    """
    def __init__(self, minSamplesSplit = 2, maxDepth = None, maxFeatures = None, min_impurity_decrease =0):
        super(RegressionTree, self).__init__(trainingFunction = varianceReduction, predictionFunc = predictionRegression, minSamplesSplit=minSamplesSplit, maxDepth = maxDepth, maxFeatures = maxFeatures, min_impurity_decrease = min_impurity_decrease)


