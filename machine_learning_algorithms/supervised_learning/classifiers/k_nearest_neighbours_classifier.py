""" This module contains code representing the k nearest neighbours algorithm
for classification """
import numpy as np
from machine_learning_algorithms.supervised_learning.base_classes.k_nearest_neighbours_base import KNearestNeighboursBase


class KNearestNeighboursClassifier(KNearestNeighboursBase):
    """ This is an implementation of the nonparametric supervised
    machine learning algorithm called k nearest neighbours, used for
    classification.

    Args:
        k:
            Number of neighbours to consider when making a prediction.
            Default to consider is 10.

        similarity_metric:
            Metric to use to determine the similarity between
            different vectors. Default is euclidean distance
            (L2 distance).

        verbose:
            Boolean value that determines whether or not to provide updates
            when the model is predicting on new examples.
    """

    def __init__(self, k=10, similarity_metric="L2", verbose=True):
        #Allow either L2 distance or L1 distance to be used
        super(KNearestNeighboursClassifier,
              self).__init__(k, similarity_metric, verbose)

    def _get_prediction(self, k_closest: np.ndarray) -> int:
        """ This method returns an integer describing the label of the
        current vector.

        Args:
            k_closest:
                Numpy array of shape (K, 2) containing the K closest vectors
                with their corresponding labels

        Returns:
            An integer representing the label of the vector 
        """
        unique, counts = np.unique(k_closest, return_counts=True)
        return unique[np.argmax(counts)]
