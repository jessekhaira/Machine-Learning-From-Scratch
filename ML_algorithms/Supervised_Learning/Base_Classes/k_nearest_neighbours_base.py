""" This module contains code for the k nearest neighbours supervised
machine learning algorithm """
import numpy as np


class KNearestNeighboursBase(object):
    """This is a minimal implementation of the nonparametric supervised
    machine learning algorithm called k-Nearest-Neighbours (kNN).

    Attributes:
        k:
            Integer representing number of neighbours to consider when making a
            prediction. Default to consider is 10.

        similarity_metric:
            String representing the metric to use to determine the similarity
            between different vectors. Default is euclidean distance (L2
            distance).

        verbose:
            Boolean value that determines whether or not to provide updates
            when the model is predicting on new examples.
    """

    def __init__(self,
                 k: int = 10,
                 similarity_metric: str = 'L2',
                 verbose: bool = True):
        #Allow either L2 distance or L1 distance to be used
        assert similarity_metric == 'L2' or similarity_metric == 'L1', (
            'Sorry, you must use either L2 distance or L1 distance')
        self.k = k
        self.similarity_metric = similarity_metric
        self.verbosity = verbose

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """This method trains the object on the feature vectors (x_train),
        and the vectors corresponding labels (y_train).

        Args:
            x_train:
                NumPy matrix of shape (M,N) where N is the number of
                features in one example, and M is the number of training
                examples.

            y_train:
                NumPy vector of shape (M,1) where M is the number of training
                examples.
        """
        # kNN just memorizes the data - the model we are "learning" is the data.
        # no need to train parameters. This is called lazy learning.
        assert x_train.shape[0] == y_train.shape[0], print(
            'x shape is (%s,%s) and y shape is (%s, %s)' %
            (x_train.shape[0], x_train.shape[1], y_train.shape[0],
             y_train.shape[1]))
        self.model_x = x_train
        self.model_y = y_train

    def predict(self, x_predict: np.ndarray) -> np.ndarray:
        """This method takes in a matrix of unlabelled vectors, and uses the
        trained kNN model to get predictions for all the vectors.

        Args:
            x_predict:
                NumPy matrix of shape (M,N) where N is the number of features
                in one example, and M is the number of examples.

        Returns:
            NumPy vector of shape (M,1) containing the labels for all the M
            examples.
        """
        assert x_predict.shape[1] == self.model_x.shape[1], (
            'The input should have the same number of features as what you \
                trained on!')
        y_pred = np.empty((x_predict.shape[0], 1))
        # Loop over every example in x_predict, get k_closest neighbours
        # and assign the prediction for this example
        for i, example in enumerate(x_predict):
            print(f'Predicting on the {i}th example, {i}/{len(x_predict)}')
            k_closest = self._get_k_closest(example, self.similarity_metric)
            y_pred[i] = self._get_prediction(k_closest)
        return y_pred

    def _get_k_closest(self, ex_x: np.ndarray, similarity: str) -> np.ndarray:
        """This method returns the k closest vectors to the current vector ex_x
        in the training set.

        Args:
            ex_x:
                NumPy vector of shape (1,N)
            similarity:
                String that is either "L2" or "L1" indicating the type of
                similarity metric being used

        Returns:
            NumPy Matrix of shape (K, 1) containing the K closest vector with
            their corresponding labels
        """
        similarity = 2 if similarity == 'L2' else 1
        distance_to_all = np.linalg.norm(self.model_x - ex_x,
                                         ord=similarity,
                                         axis=1,
                                         keepdims=True)
        assert distance_to_all.shape == (self.model_x.shape[0], 1), (
            'You should have a distance vector of shape (m,1)!')
        return self.model_y[np.argpartition(distance_to_all, self.k,
                                            axis=0)][:self.k]

    def _get_prediction(self, k_closest: np.ndarray):
        """This method is the only place where the kNN regressor and classifier
        differ.

        Classifier -> take the most common class among the neighbours.
        Regressor -> take the average among the neighbours.

        Thus, this method will be left as an abstract method, to be inherited by
        both the kNN classifier and regressor and overriden with their own
        specific implementations.
        
        k_closest:
            A numpy array which represents the k closest neigbours to the current
            vector 
        """
        raise NotImplementedError
