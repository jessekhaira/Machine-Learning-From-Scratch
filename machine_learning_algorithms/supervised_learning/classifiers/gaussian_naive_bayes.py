""" This module contains code representing the gaussian naive bayes
supervised machine learning algorithm """
import numpy as np
import math
from typing import Tuple


class BaseNaiveBayes(object):

    def _get_probability_classes(
            self, ytrain: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        counts_each_class = np.unique(ytrain, return_counts=True)
        # class probabilties aree simply the frequency of instances that
        # belong to each class divided by the total number of instances
        unique_lables = counts_each_class[0]
        class_probabilities = counts_each_class[1] / ytrain.shape[1]
        return unique_lables, class_probabilities

    def _get_probability_pxy(self, xtrain: np.ndarray, ytrain: np.ndarray):
        raise NotImplementedError


class GaussianNaiveBayes(BaseNaiveBayes):
    """ This class represents the gaussian naive bayes classifier. Naive
    bayes is a probabilistic classifier that produces a probabiliity
    distribution over the classes in the dataset, with a (naive)
    assumption that all the input features are independent.

    Naive Bayes is an example of a generative algorithm that can
    perform classification through learning the joint probability
    distribution P(Y,X) = P(Y)P(X|Y) and then using that to compute
    P(Y|X) using Bayes Theorem.

    P(Y|X) = P(Y,X)/P(X)

    P(X) is dropped because we don't care about getting the correct
    probabilities, we just care that the highest probability is assigned
    to the correct class, which it will be with or without the normalizing
    factor of the marginal probability P(X).
    """

    def __init__(self):
        # need to store P(Y) and the unique labels
        self.Y = None
        self.class_probabilities = None
        # need to store P(X|Y)
        self.PX_Y_mean = None
        self.PX_Y_std = None

    def fit(self, xtrain, ytrain):
        """
        This method trains the classifier on the dataset. Fitting is fast fitting is fast -  as there is no need 
        to carry out expensive optimization because we aren't trying to find coefficients.

        The fitting procedure is just computing P(Y) for the class labels, and then P(X|Y) for all the class labels,
        and storing them.

        Parameters:
        -> xtrain (NumPy matrix): Matrix of shape (features, examples)
        -> ytrain (NumPy vector): Vector of shape (1, examples)

        Returns:
        -> None 
        """
        self.Y, self.class_probabilities = self._get_probability_classes(ytrain)
        self._get_probability_pxy(xtrain, ytrain)

    def _get_probability_pxy(self, xtrain, ytrain):
        # GAUSSIAN naive bayes - not going to get P(X|Y) by frequencies as our features are continous random variables
        # instead we assume the features are all normally distributeed
        # and then compute probabilities using the gaussian PDF

        # thus, we need the mean and std dev for P(X|Y=class 1), P(X|Y=class 2) etc

        trainMatrix = np.hstack((xtrain.T, ytrain.T))
        self.PX_Y_mean = np.zeros((xtrain.shape[0], len(self.Y)))
        self.PX_Y_std = np.zeros((xtrain.shape[0], len(self.Y)))
        for label_y in self.Y:
            label_y = int(label_y)
            idxs_filtered = np.where(trainMatrix[:, -1] == label_y)
            filteredMatrix = trainMatrix[idxs_filtered[0], :-1].T
            mean_vectorFeatures_labelY = np.mean(filteredMatrix, axis=1)
            std_devVecFeatures_labelY = np.std(filteredMatrix, axis=1)
            self.PX_Y_mean[:, label_y] = mean_vectorFeatures_labelY
            self.PX_Y_std[:, label_y] = std_devVecFeatures_labelY

    def predict(self, X):
        """
        This method carries out classification using Bayes Theorem. Since this is a Gaussian Naive Bayes,
        the probability for each class is computed using the PDF for a normal distribution.

        Parameters:
        -> X (single NumPy vector, or NumPy matrix): Matrix of examples to predict on of shape (features, examples)

        Returns:
        -> Output (NumPy vector): Predictions for every vector in the input  
        """
        predictions = np.zeros((1, X.shape[1]))
        for i in range(X.shape[1]):
            vector = X[:, i].reshape(-1, 1)
            highestProb = float('-inf')
            predictedClass = -1
            for y_label in self.Y:
                y_label = int(y_label)
                # compute P(Y|X) for the specific feature vals we have
                PY_X = np.log(self.class_probabilities[y_label] + 1e-10)
                for feature_row in range(vector.shape[0]):
                    x = vector[feature_row]
                    PX_Y = self._computeProbability(x, feature_row, y_label)
                    # this is where naive part of naive bayes comes in
                    # we use all our features and combine them as such:
                    # P(Y|X1,X2,X3,..,XN) = P(X1|Y)*P(X2|Y)*P(X3|Y)*...*P(XN|Y)*P(Y)
                    # IE assume all our features are independent. This is vulnerable to numeric
                    # underflows so use ln(prob) instead and add
                    PY_X += (np.log(PX_Y[0] + 1e-10))
                if PY_X > highestProb:
                    highestProb = PY_X
                    predictedClass = y_label
            predictions[:, i] = predictedClass

        return predictions

    def _computeProbability(self, feature_value, feature_row, label):
        eps = 1e-8
        # you can compute the PDF all in one
        # Just find it easier to seperate out the terms and then combine at the end cause theres
        # so much going on
        denominator = 1 / (
            self.PX_Y_std[feature_row, label] * np.sqrt(2 * math.pi) + eps)
        exp_termNumerator = feature_value - self.PX_Y_mean[feature_row, label]
        exp_termDenominator = self.PX_Y_std[feature_row, label] + eps
        combined = np.power(exp_termNumerator / exp_termDenominator, 2)
        exp_term = np.exp(-0.5 * combined)
        return denominator * exp_term
