""" This module contains code representing the regression decision
tree algorithm """
from machine_learning_algorithms.supervised_learning.base_classes.decision_tree import BaseDecisionTree
from machine_learning_algorithms.utility.decision_tree_functions import prediction_regression
from machine_learning_algorithms.utility.decision_tree_functions import variance_reduction


class RegressionTree(BaseDecisionTree):
    """ This class represents the regression decision tree algorithm,
    using the variance reduction cost function.

    Attributes:
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
                 minSamplesSplit=2,
                 maxDepth=None,
                 maxFeatures=None,
                 min_impurity_decrease=0):
        super(RegressionTree,
              self).__init__(trainingFunction=variance_reduction,
                             prediction_func=prediction_regression,
                             minSamplesSplit=minSamplesSplit,
                             maxDepth=maxDepth,
                             maxFeatures=maxFeatures,
                             min_impurity_decrease=min_impurity_decrease)
