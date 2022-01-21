""" This module contains miscellaneous utility functions used
throughout the package"""
import numpy as np


def one_hot_encode(y: np.ndarray) -> np.ndarray:
    """ This functions purpose is to create a (m, n) matrix
    where each column in the matrix contains a one hot encoded
    vector created from the input vector y, where the number of unique
    values inside of the input vector y determine the number of columns
    inside the output matrix.

    Typically used to create the label matrix when training a
    multiclass classification algorithm """
    # Squish y to be one row
    y_use = y.reshape(-1)
    num_labels = len(np.unique(y_use))
    num_examples = len(y_use)
    output_matrix = np.zeros((num_examples, num_labels))
    output_matrix[np.arange(num_examples), y_use] = 1
    return output_matrix.T


def one_hot_encode_feature(num_features: int, idx_one: int) -> np.ndarray:
    """ This functions purpose is to create a one hot encoded vector
    of shape (n, 1) where n is the num_feature argument, and the value
    that has a 1 is determined by the idx_one argumnet """
    vector = np.zeros((num_features, 1))
    vector[idx_one] = 1
    return vector


def convert_to_highest_pred(arr: np.ndarray) -> np.ndarray:

    arr = np.argmax(arr, axis=0)
    return arr


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """ This function computes the euclidean distance between
    two N dimensional vectors"""
    # euclidean distance is the l2 norm of the vector x- y
    return np.linalg.norm(x - y, ord=2)


def find_row_column_max_element(tensor: np.ndarray):
    idxs = np.unravel_index(np.nanargmax(tensor), tensor.shape)
    return idxs


def gradient_clipping(dparams: np.ndarray) -> None:
    for gradient_tensor in dparams:
        np.clip(gradient_tensor, -5, 5, out=gradient_tensor)


def get_unique_chars(text_file):
    """ This function produces a basic vocabulary for a text file
    through tokenizing by splitting on white space, and returning
    a set of all the tokens found """
    return list(set(text_file))


def map_idx_to_char(chars):
    """ This function is often used during the preprocessing phase
    in natural language processing to map each of the tokens inside
    of the chars argument to an index in a vocabulary """
    return {i: char for i, char in enumerate(chars)}


def map_char_to_idx(chars):
    return {char: idx for idx, char in enumerate(chars)}


def get_covariance_matrix(matrix: np.ndarray) -> np.ndarray:
    mean_features = np.mean(matrix, axis=1, keepdims=True)
    # vectorize operation to get covariance matrix - don't want to do
    # an expensive python for loop
    num_examples = matrix.shape[1]
    return 1 / (num_examples - 1) * (matrix - mean_features).dot(
        (matrix - mean_features).T)
