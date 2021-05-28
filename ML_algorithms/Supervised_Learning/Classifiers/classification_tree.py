""" This module contains code representing the classification decision
tree algorithm """
from ML_algorithms.Supervised_Learning.Base_Classes.DecisionTree import BaseDecisionTree
from ML_algorithms.Utility.DecisionTreeFunctions import entropyGain
from ML_algorithms.Utility.DecisionTreeFunctions import giniGain
from ML_algorithms.Utility.DecisionTreeFunctions import predictionClassification
from typing import Union


class ClassificationTree(BaseDecisionTree):
    """ This class implements the BaseDecisionTree class for
    the purposes of performing classification.

    Attributes:
        entropy:
            Boolean value indicating whether to train using entropy based
            information gain.

        minSamplesSplit:
            Integer indicating the minimum number of examples that have
            to fall in this node to justify splitting further

        maxDepth:
            Integer representing the maximum depth to grow this tree,
            or None if not wanted

        maxFeatures:
            Integer representing the maximum number of features to use
            to determine the split, or None if not wanted

        min_impurity_decrease:
            Integer representing the minimum decrease in impurity to justify
            splitting a node
    """

    def __init__(self,
                 entropy: bool = True,
                 minSamplesSplit: int = 2,
                 maxDepth: Union[None, int] = None,
                 maxFeatures: Union[None, int] = None,
                 min_impurity_decrease: int = 0):
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
