""" This module contains code representing the k nearest neighbours
algorithm implemented to perform regression """
import numpy as np
from ML_algorithms.Supervised_Learning.Base_Classes.k_nearest_neighbours_base import KNearestNeighboursBase


class KNearestNeighboursRegressor(KNearestNeighboursBase):
    """ This is a minimal implementation of the nonparametric supervised
    machine learning algorithm called k nearest neighbours(kNN) for
    regression.

    Attributes:
        k:
            Integer representing the number of neighbours to consider
            when making a prediction. Default to consider is 10.

        similarity_metric:
            String representing the metric to use to determine the
            similarity between vectors. Default is euclidean
            distance (L2 distance).

        verbose:
            Boolean value that determines whether or not to provide updates
            when the model is predicting on new examples.
    """

    def __init__(self,
                 k: int = 10,
                 similarity_metric: str = "L2",
                 verbose: bool = True):
        #Allow either L2 distance or L1 distance to be used
        super(KNearestNeighboursRegressor,
              self).__init__(k, similarity_metric, verbose)

    def _get_prediction(self, k_closest):
        """
        This method returns a float for the numeric label of the current vector.

        Input:
        k_closest (NumPy matrix) -> Matrix of shape (K, 2) containing the K closest vectors
        with their corresponding labels 

        Output (float) -> label of the vector 
        """
        return np.mean(k_closest)