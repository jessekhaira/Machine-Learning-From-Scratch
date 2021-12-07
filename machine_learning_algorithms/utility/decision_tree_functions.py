import numpy as np
from typing import Tuple
from machine_learning_algorithms.utility.score_functions import total_sum_of_squares
from machine_learning_algorithms.utility.score_functions import entropy
from machine_learning_algorithms.utility.score_functions import gini_index


def prediction_classification(labels):
    # Just predict the most commonly occurring class AKA the mode of the labels
    vals, counts = np.unique(labels, return_counts=True)
    idx = np.argmax(counts)
    return vals[idx]


def prediction_regression(labels: np.ndarray) -> float:
    # Just predict the average of the values that fall in this leaf!
    return np.mean(labels)


def entropy_gain(root, left, right) -> float:
    entropy_root = entropy(root)
    entropy_left_node = entropy(left)
    entropy_right_node = entropy(right)
    num_examples_left = left.shape[1]
    num_examples_right = right.shape[1]
    fraction_of_data_left, fraction_of_data_right = get_fractions(
        num_examples_left, num_examples_right)
    assert (fraction_of_data_left + fraction_of_data_right) == 1, (
        "Somethings wrong with how your data is splitting into " +
        "left and right datasets")
    # Intuitively, we want a feature that splits the data perfectly into
    # pure nodes on the left and right side, meaning that going from
    # the root node to the left nodes and right nodes, we gain a
    # lot of information
    return entropy_root - (fraction_of_data_left * entropy_left_node +
                           fraction_of_data_right * entropy_right_node)


def get_fractions(num_examples_left: int,
                  num_examples_right: int) -> Tuple[float, float]:
    fraction_left = num_examples_left / (num_examples_left + num_examples_right)
    fraction_right = 1 - fraction_left
    return fraction_left, fraction_right


def gini_gain(root, left, right):
    gini_curr = gini_index(root)
    gini_left_node = gini_index(left)
    gini_right_node = gini_index(right)
    num_examples_left = left.shape[1]
    num_examples_right = right.shape[1]
    fraction_left, fraction_right = get_fractions(num_examples_left,
                                                  num_examples_right)
    assert (fraction_left + fraction_right) == 1, (
        "Somethings  wrong with how your data is splitting into " +
        "left and right datasets")
    return gini_curr - (fraction_left * gini_left_node +
                        fraction_right * gini_right_node)


def varianceReduction(root, left, right):
    # In a regression tree, at any node, the expected value of all of the examples that fall in the node
    # IS the prediction. So getting the variance is like calculating the RSS, except our prediction for every
    # example is the same of the mean value
    varianceRoot = total_sum_of_squares(root)
    varianceLeft = total_sum_of_squares(left)
    varianceRight = total_sum_of_squares(right)
    num_examples_left = left.shape[1]
    num_examples_right = right.shape[1]
    fraction_left, fraction_right = get_fractions(num_examples_left,
                                                  num_examples_right)
    assert (
        fraction_left + fraction_right
    ) == 1, "Somethings  wrong with how your data is splitting into left and right datasets"
    # Ideally you have 0 variance in left node and 0 variance in right node since your predictions are just perfect! :D
    return varianceRoot - (fraction_left * varianceLeft +
                           fraction_right * varianceRight)
