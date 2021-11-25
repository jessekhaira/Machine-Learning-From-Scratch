""" This module contains score functions used to assess
the performance of a machine learning algorithm performing
supervised learning """
import numpy as np


def accuracy(ylabel: np.ndarray, yhat: np.ndarray) -> float:
    return np.mean(ylabel == yhat)


def mean_squared_error(ylabel: np.ndarray, yhat: np.ndarray) -> float:
    return np.square(np.subtract(ylabel, yhat)).mean()


def mean_absolute_error(ylabel: np.ndarray, yhat: np.ndarray) -> float:
    return np.abs(ylabel - yhat).mean()


def root_mean_squared_error(ylabel: np.ndarray, yhat: np.ndarray) -> float:
    output = mean_squared_error(ylabel, yhat)
    return output**0.5


def r_squared(ylabel: np.ndarray, yhat: np.ndarray) -> float:
    calced_RSS = RSS(ylabel, yhat)
    calced_TSS = TSS(ylabel)
    return 1 - (calced_RSS / calced_TSS)


def r_squared_adjusted(ylabel: np.ndarray, yhat: np.ndarray,
                       num_features: int) -> float:
    """ You can just add more polynomial features and fit a super complex
    function with linear regression models to inflate the R2 score
    [ie. overfitting]. This score gives a better representation of how
    well your model explains the variance in the response variable """
    num_examples = ylabel.shape[1]
    adjusted_residual_sum_of_squares = RSS(
        ylabel, yhat) / (num_examples - num_features - 1)
    adjusted_total_sum_of_squares = TSS(ylabel) / (num_examples - 1)
    return 1 - (adjusted_residual_sum_of_squares /
                adjusted_total_sum_of_squares)


def RSS(ylabel, yhat):
    # Find the sum of all the residuals over every single example, not averaged over the examples
    return np.sum(np.square(np.subtract(ylabel, yhat)))


def TSS(ylabel):
    # Get the total variance in the response variable, not averaged over the examples
    labels_mean = np.mean(ylabel)
    return np.sum(np.square(np.subtract(ylabel, labels_mean)))


def entropy(labels):
    probabilityEachClass = getCounts(labels)
    return -1 * sum(probabilityEachClass[i] * np.log(probabilityEachClass[i])
                    for i in probabilityEachClass)


def giniIndex(labels):
    probabilityEachClass = getCounts(labels)
    sumEachSquared = sum(
        probabilityEachClass[i]**2 for i in probabilityEachClass)
    return 1 - sumEachSquared


def getCounts(labels):
    unique, counts = np.unique(labels, return_counts=True)
    totalCount = np.sum(counts)
    probabilityEachClass = dict(zip(unique, counts / totalCount))
    return probabilityEachClass