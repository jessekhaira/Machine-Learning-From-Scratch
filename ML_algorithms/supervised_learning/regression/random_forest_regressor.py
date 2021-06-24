""" This module contains code for the supervised ensemble machine learning
algorithm random forests, implemented specifically for regression """
from ML_algorithms.supervised_learning.base_classes.BaggedForest import BaggedForest


class RandomForestRegressor(BaggedForest):
    """ This class represents the random forest algorithm for the task
    of regression.

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
                 num_estimators=100,
                 max_samples=None,
                 bootstrap=False,
                 minSamplesSplit=2,
                 maxDepth=None,
                 maxFeatures=None,
                 min_impurity_decrease=0,
                 verbose=False):
        super(RandomForestRegressor,
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
