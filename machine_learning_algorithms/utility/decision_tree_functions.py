"""This module contains a bunch of utility functions
related to building and using decision trees"""
import numpy as np
from typing import Tuple
from machine_learning_algorithms.utility.score_functions import total_sum_of_squares
from machine_learning_algorithms.utility.score_functions import entropy
from machine_learning_algorithms.utility.score_functions import gini_index
from machine_learning_algorithms.supervised_learning.base_classes.decision_tree import DecisionTreeNode


def prediction_classification(labels: np.ndarray) -> int:
    """ This function is meant to be used to predict a label at a
    leaf node inside of a classification tree, which will be
    the most commonly occurring label inside this node"""
    # Just predict the most commonly occurring class AKA the mode of the labels
    vals, counts = np.unique(labels, return_counts=True)
    idx = np.argmax(counts)
    return vals[idx]


def prediction_regression(labels: np.ndarray) -> float:
    """ This function is meant to be used to predict a label at a
    leaf node inside of a regression tree, which will be
    the average of all the labels inside this node"""
    # Just predict the average of the values that fall in this leaf!
    return np.mean(labels)


def entropy_gain(root: DecisionTreeNode, left: DecisionTreeNode,
                 right: DecisionTreeNode) -> float:
    """ This function represents the information gain criterion 
    based on entropy, used to train decision trees by helping choose
    which feature + split point should be used within a given node"""
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


def gini_gain(root: DecisionTreeNode, left: DecisionTreeNode,
              right: DecisionTreeNode) -> float:
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


def variance_reduction(root: DecisionTreeNode, left: DecisionTreeNode,
                       right: DecisionTreeNode) -> float:
    """This function represents the variance reduction algorithm
    used to train decision trees performing regression, helping
    to choose the optimal feature + split point pair to split
    a node"""
    # In a regression tree, at any node, the expected value of all
    # of the examples that fall in the node IS the prediction. So
    # getting the variance is like calculating the RSS, except our
    # prediction for every example is the same of the mean value
    variance_root = total_sum_of_squares(root)
    variance_left = total_sum_of_squares(left)
    variance_right = total_sum_of_squares(right)
    num_examples_left = left.shape[1]
    num_examples_right = right.shape[1]
    fraction_left, fraction_right = get_fractions(num_examples_left,
                                                  num_examples_right)
    assert (fraction_left + fraction_right) == 1, (
        "Somethings wrong with how your data is splitting into " +
        "left and right datasets")
    # Ideally you have 0 variance in left node and 0 variance in right
    # node since your predictions are just perfect! :D
    return variance_root - (fraction_left * variance_left +
                            fraction_right * variance_right)
