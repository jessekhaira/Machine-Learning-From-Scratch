""" This module contains code representing the bagged forest ensemble
classification algorithm """
from machine_learning_algorithms.supervised_learning.base_classes.BaggedForest import BaggedForest
from typing import Literal, Union


class BaggedForestClassifier(BaggedForest):
    """ This class represents bootstrap aggregated (bagged) decision trees
    performing the task of classification.

    Attributes:
        criterion:
            String representing the objective function to use. Should be either
            "gini" or "entropy.

        num_estimators:
            Integer representing the number of estimators to include in the
            ensemble

        max_samples:
            None, integer, or floating point value representing the number of
            samples to put into each bootstrap sample. If None, the num sample
            is equal to the dataset size. If int, the number of samples is
            equal to the size of the int. If float, the float should be between
            0 and 1 and it will indicate the percentage to include in each
            bootstrap sample.

        bootstrap:
            Boolean indicating whether to bootstrap the dataset when building
            the ensemble.

        minSamplesSplit:
            Integer indicating the minimum number of examples that have to fall
            in this node to justify splitting further

        maxDepth:
            Integer representing the maximum depth to grow this tree, or None
            if this hyperparameter isn't used.

        maxFeatures:
            Integer representing the maximum number of features to use to
            determine the split, or None if this hyperparameter isn't
            used.

        min_impurity_decrease:
            Floating point value representing the minimum decrease in impurity
            to justify splitting the node

        verbose:
            Boolean indicating whether to give updates when fitting and
            predicting
    """

    def __init__(self,
                 criterion: Literal["gini", "entropy"],
                 num_estimators: int = 100,
                 max_samples: Union[None, int, float] = None,
                 bootstrap: bool = False,
                 minSamplesSplit: int = 2,
                 maxDepth: Union[int, None] = None,
                 maxFeatures: Union[int, None] = None,
                 min_impurity_decrease: float = 0,
                 verbose: bool = False):
        super(BaggedForestClassifier,
              self).__init__(typeSupervised=0,
                             criterion=criterion,
                             num_estimators=num_estimators,
                             max_samples=max_samples,
                             bootstrap=bootstrap,
                             minSamplesSplit=minSamplesSplit,
                             maxDepth=maxDepth,
                             maxFeatures=maxFeatures,
                             min_impurity_decrease=min_impurity_decrease,
                             verbose=verbose)
