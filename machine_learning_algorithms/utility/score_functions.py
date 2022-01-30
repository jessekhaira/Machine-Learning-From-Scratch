""" This module contains score functions used to assess
the performance of a machine learning algorithm performing
supervised learning """
import numpy as np


def accuracy(ylabel: np.ndarray, yhat: np.ndarray) -> float:
    return np.mean(ylabel == yhat)


def mean_squared_error(ylabel: np.ndarray, yhat: np.ndarray) -> float:
    """ This function represents the mean squared error function, often
    used to assess the performance of machine learning algorithms
    performing regression, and also used as the objective function to train
    regression algorithms iteratively"""
    return np.square(np.subtract(ylabel, yhat)).mean()


def mean_absolute_error(ylabel: np.ndarray, yhat: np.ndarray) -> float:
    """ This function represents the mean absolute error function
    often used as a metric to assess the performance of a machine
    learning algorithm performing regression"""
    return np.abs(ylabel - yhat).mean()


def root_mean_squared_error(ylabel: np.ndarray, yhat: np.ndarray) -> float:
    """ This function computes the root mean squared error, a function often
    used as a metric to assess the performance of a machine learning
    algorithm performing regression"""
    output = mean_squared_error(ylabel, yhat)
    return output**0.5


def r_squared(ylabel: np.ndarray, yhat: np.ndarray) -> float:
    """ This function computes the r squared metric, typically used to
    assess the performance of machine learning algorithms performing
    regression """
    residual_sum_of_squares_val = residual_sum_of_squares(ylabel, yhat)
    total_sum_of_squares_val = total_sum_of_squares(ylabel)
    return 1 - (residual_sum_of_squares_val / total_sum_of_squares_val)


def r_squared_adjusted(ylabel: np.ndarray, yhat: np.ndarray,
                       num_features: int) -> float:
    """ You can just add more polynomial features and fit a super complex
    function with linear regression models to inflate the R2 score
    [ie. overfitting]. This score gives a better representation of how
    well your model explains the variance in the response variable """
    num_examples = ylabel.shape[1]
    adjusted_residual_sum_of_squares = residual_sum_of_squares(
        ylabel, yhat) / (num_examples - num_features - 1)
    adjusted_total_sum_of_squares = total_sum_of_squares(ylabel) / (
        num_examples - 1)
    return 1 - (adjusted_residual_sum_of_squares /
                adjusted_total_sum_of_squares)


def residual_sum_of_squares(ylabel: np.ndarray, yhat: np.ndarray) -> float:
    """ This function represents the residual sum of squares score function
    used often to assess the performance of a machine learning algorithm
    performing regression, and also used as the objective function to train
    regression algorithms"""
    return np.sum(np.square(np.subtract(ylabel, yhat)))


def total_sum_of_squares(ylabel: np.ndarray) -> float:
    """ Get the total variance in the response variable, not averaged
    over the examples """
    labels_mean = np.mean(ylabel)
    return np.sum(np.square(np.subtract(ylabel, labels_mean)))


def entropy(labels: np.ndarray) -> float:
    """ This function computes the entropy of the input tensor,
    which should be a vector of probabilities with every value
    between 0 and 1 and the sum of the vector equating to 1"""
    probability_each_class = get_counts(labels)
    return -1 * sum(
        probability_each_class[i] * np.log(probability_each_class[i])
        for i in probability_each_class)


def gini_index(labels: np.ndarray) -> float:
    """ This function computes the gini index of the input tensor."""
    probability_each_class = get_counts(labels)
    sum_each_squared = sum(
        probability_each_class[i]**2 for i in probability_each_class)
    return 1 - sum_each_squared


def get_counts(labels: np.ndarray):
    unique, counts = np.unique(labels, return_counts=True)
    total_count = np.sum(counts)
    probability_each_class = dict(zip(unique, counts / total_count))
    return probability_each_class
