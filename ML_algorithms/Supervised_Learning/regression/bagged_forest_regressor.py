""" This module contains code representing an implementation of the
bagged forest algorithm for regression """
from ML_algorithms.Supervised_Learning.Base_Classes.BaggedForest import BaggedForest
from typing import Union


class BaggedForestRegression(BaggedForest):
    """ This class represents bootstrap aggregated (bagged) decision trees
   performing the task of regression.

    Attributes:
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
                 num_estimators: int = 100,
                 max_samples: Union[int, float, None] = None,
                 bootstrap: bool = False,
                 minSamplesSplit: int = 2,
                 maxDepth: int = None,
                 maxFeatures: Union[int, None] = None,
                 min_impurity_decrease: float = 0,
                 verbose: bool = False):

        super(BaggedForestRegression,
              self).__init__(typeSupervised=1,
                             criterion=None,
                             num_estimators=num_estimators,
                             max_samples=max_samples,
                             bootstrap=bootstrap,
                             minSamplesSplit=minSamplesSplit,
                             maxDepth=maxDepth,
                             maxFeatures=maxFeatures,
                             min_impurity_decrease=min_impurity_decrease,
                             verbose=verbose)
