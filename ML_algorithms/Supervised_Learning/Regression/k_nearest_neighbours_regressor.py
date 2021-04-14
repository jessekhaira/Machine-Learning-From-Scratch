import numpy as np
from ML_algorithms.Supervised_Learning.Base_Classes.k_nearest_neighbours_base import KNearestNeighboursBase


class KNearestNeighboursRegressor(KNearestNeighboursBase):
    """
    This is a minimal implementation of the nonparametric supervised
    machine learning algorithm called kNN, for regression. 

    Params:

    k -> Number of neighbours to consider when making a prediction. 
    Default to consider is 10.

    similarity_metric -> Metric to use to determine the similarity between 
    different vectors. Default is euclidean distance (L2 distance).

    verbose -> Boolean value that determines whether or not to provide updates
    when the model is predicting on new examples.   
    """

    def __init__(self, k=10, similarity_metric="L2", verbose=True):
        #Allow either L2 distance or L1 distance to be used
        super(k_Nearest_Neighbours, self).__init__(k, similarity_metric,
                                                   verbose)

    def _getPrediction(self, k_closest):
        """
        This method returns a float for the numeric label of the current vector.

        Input:
        k_closest (NumPy matrix) -> Matrix of shape (K, 2) containing the K closest vectors
        with their corresponding labels 

        Output (float) -> label of the vector 
        """
        return np.mean(k_closest)