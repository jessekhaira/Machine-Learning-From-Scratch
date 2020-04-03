from collections import Counter
import numpy as np 
import sys
sys.path.append("/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Supervised_Learning/Utils")
from kNearestNeighbours_baseClass import kNearestNeighbours_base

class k_Nearest_Neighbours(kNearestNeighbours_base):
    """
    This is a minimal implementation of the nonparametric supervised
    machine learning algorithm called kNN, used for classification. 

    Params:

    k -> Number of neighbours to consider when making a prediction. 
    Default to consider is 10.

    similarity_metric -> Metric to use to determine the similarity between 
    different vectors. Default is euclidean distance (L2 distance).
    """
    def __init__(self, k = 10, similarity_metric = "L2", verbose = True):
        #Allow either L2 distance or L1 distance to be used
        super(k_Nearest_Neighbours, self).__init__(k, similarity_metric, verbose)

    def _getPrediction(self, k_closest):
        """
        This method returns an integer describing the label of the current vector.

        Input:
        k_closest (NumPy matrix) -> Matrix of shape (K, 2) containing the K closest vectors
        with their corresponding labels 

        Output (int) -> label of the vector 
        """
        unique, counts = np.unique(k_closest, return_counts=True)
        return unique[np.argmax(counts)]