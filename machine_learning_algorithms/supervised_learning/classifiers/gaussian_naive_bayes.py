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

    def _get_probability_x_conditioned_y(self, xtrain: np.ndarray,
                                         ytrain: np.ndarray):
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
        self.y = None
        self.class_probabilities = None
        # need to store P(X|Y)
        self.p_x_conditioned_y_mean = None
        self.p_x_conditioned_y_std = None

    def fit(self, xtrain: np.ndarray, ytrain: np.ndarray) -> None:
        """ This method trains the classifier on the dataset. Fitting
        is fast, as there is no need to carry out expensive optimization
        because we aren't trying to fit coefficients.

        The fitting procedure is just computing P(Y) for the class labels,
        and then P(X|Y) for all the class labels, and storing them.

        Args:
            xtrain:
                Numpy array of shape (features, examples) representing the
                training vectors for the algorithm

            ytrain:
                Numpy array of shape (1, examples) representing the labels for
                the training vectors for the algorithm
        """
        self.y, self.class_probabilities = self._get_probability_classes(ytrain)
        self._get_probability_x_conditioned_y(xtrain, ytrain)

    def _get_probability_x_conditioned_y(self, xtrain: np.ndarray,
                                         ytrain: np.ndarray) -> None:
        """ GAUSSIAN naive bayes - not going to get P(X|Y) by frequencies
        as our features are continous random variables. Instead, we
        assume the features are all normally distributed, and then
        compute probabilities using the gaussian PDF. Thus, we
        need the mean and std dev for P(X|Y=class 1), P(X|Y=class 2) etc
        """

        train_matrix = np.hstack((xtrain.T, ytrain.T))
        self.p_x_conditioned_y_mean = np.zeros((xtrain.shape[0], len(self.y)))
        self.p_x_conditioned_y_std = np.zeros((xtrain.shape[0], len(self.y)))
        for label_y in self.y:
            label_y = int(label_y)
            idxs_filtered = np.where(train_matrix[:, -1] == label_y)
            filtered_matrix = train_matrix[idxs_filtered[0], :-1].T
            p_x_conditioned_y_mean = np.mean(filtered_matrix, axis=1)
            p_x_conditioned_y_std = np.std(filtered_matrix, axis=1)
            self.p_x_conditioned_y_mean[:, label_y] = p_x_conditioned_y_mean
            self.p_x_conditioned_y_std[:, label_y] = p_x_conditioned_y_std

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        This method carries out classification using Bayes Theorem. Since this is a Gaussian Naive Bayes,
        the probability for each class is computed using the PDF for a normal distribution.

        Parameters:
        -> x (single NumPy vector, or NumPy matrix): Matrix of examples to predict on of shape (features, examples)

        Returns:
        -> Output (NumPy vector): Predictions for every vector in the input  
        """
        predictions = np.zeros((1, x.shape[1]))
        for i in range(x.shape[1]):
            vector = x[:, i].reshape(-1, 1)
            highest_prob = float('-inf')
            predicted_class = -1
            for y_label in self.y:
                y_label = int(y_label)
                # compute P(Y|x) for the specific feature vals we have
                p_y_conditioned_x = np.log(self.class_probabilities[y_label] +
                                           1e-10)
                for feature_row in range(vector.shape[0]):
                    x = vector[feature_row]
                    p_x_conditioned_y = self._computeProbability(
                        x, feature_row, y_label)
                    # this is where naive part of naive bayes comes in
                    # we use all our features and combine them as such:
                    # P(Y|X1,X2,X3,..,XN) = P(X1|Y)*P(X2|Y)*...*P(XN|Y)*P(Y)
                    # IE assume all our features are independent. This is
                    # vulnerable to numeric underflows so use ln(prob)
                    # instead and add together
                    p_y_conditioned_x += (np.log(p_x_conditioned_y[0] + 1e-10))
                if p_y_conditioned_x > highest_prob:
                    highest_prob = p_y_conditioned_x
                    predicted_class = y_label
            predictions[:, i] = predicted_class

        return predictions

    def _computeProbability(self, feature_value, feature_row, label):
        eps = 1e-8
        # you can compute the PDF all in one
        # Just find it easier to seperate out the terms and then combine at the end cause theres
        # so much going on
        denominator = 1 / (self.p_x_conditioned_y_std[feature_row, label] *
                           np.sqrt(2 * math.pi) + eps)
        exp_termNumerator = feature_value - self.p_x_conditioned_y_mean[
            feature_row, label]
        exp_termDenominator = self.p_x_conditioned_y_std[feature_row,
                                                         label] + eps
        combined = np.power(exp_termNumerator / exp_termDenominator, 2)
        exp_term = np.exp(-0.5 * combined)
        return denominator * exp_term
