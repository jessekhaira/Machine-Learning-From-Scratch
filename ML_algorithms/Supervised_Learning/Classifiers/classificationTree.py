import numpy as np
from ML_algorithms.Supervised_Learning.Base_Classes.DecisionTree import BaseDecisionTree
from ML_algorithms.Utility.DecisionTreeFunctions import entropyGain
from ML_algorithms.Utility.DecisionTreeFunctions import giniGain
from ML_algorithms.Utility.DecisionTreeFunctions import predictionClassification


class ClassificationTree(BaseDecisionTree):
    """
    This class represents a classification decision tree trained using either entropy based information gain or
    gini based information gain.

    Parameters:
    -> entropy (boolean): Boolean value indicating whether to train using entropy based information gain. 

    -> gini (boolean): Boolean value indicating whether to train using gini based information gain. 

    -> minSamplesSplit (int): Integer indicating the minimum number of examples that have to fall in this node to justify splitting further

    -> maxDepth (int): Integer representing the maximum depth to grow this tree 
    
    -> maxFeatures (int): Integer representing the maximum number of features to use to determine the split

    -> min_impurity_decrease (int): The minimum decrease in impurity to justify splitting a node 
    """

    def __init__(self,
                 entropy=True,
                 minSamplesSplit=2,
                 maxDepth=None,
                 maxFeatures=None,
                 min_impurity_decrease=0):
        if entropy:
            trainFunction = entropyGain
        else:
            trainFunction = giniGain
        predictionFunc = predictionClassification
        super(ClassificationTree,
              self).__init__(trainingFunction=trainFunction,
                             predictionFunc=predictionFunc,
                             minSamplesSplit=minSamplesSplit,
                             maxDepth=maxDepth,
                             maxFeatures=maxFeatures,
                             min_impurity_decrease=min_impurity_decrease)
