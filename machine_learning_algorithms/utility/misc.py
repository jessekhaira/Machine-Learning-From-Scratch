""" This module contains miscellaneous utility functions used
throughout the package"""
import numpy as np


def one_hot_encode(y: np.ndarray) -> np.ndarray:
    # Squish y to be one row
    y_use = y.reshape(-1)
    num_labels = len(np.unique(y_use))
    num_examples = len(y_use)
    output_matrix = np.zeros((num_examples, num_labels))
    output_matrix[np.arange(num_examples), y_use] = 1
    return output_matrix.T


def one_hot_encode_feature(numFeatures, idxOne):
    vector = np.zeros((numFeatures, 1))
    vector[idxOne] = 1
    return vector


def convert_to_highest_pred(arr: np.ndarray):
    arr = np.argmax(arr, axis=0)
    return arr


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    # euclidean distance is the l2 norm of the vector x- y
    return np.linalg.norm(x - y, ord=2)


def find_row_column_max_element(tensor: np.ndarray):
    idxs = np.unravel_index(np.nanargmax(tensor), tensor.shape)
    return idxs


def gradient_clipping(dparams: np.ndarray) -> None:
    for gradient_tensor in dparams:
        np.clip(gradient_tensor, -5, 5, out=gradient_tensor)


def get_unique_chars(txtFile):
    return list(set(txtFile))


def map_idx_to_char(chars):
    return {i: char for i, char in enumerate(chars)}


def map_char_to_idx(chars):
    return {char: idx for idx, char in enumerate(chars)}


def get_covariance_matrix(matrix):
    mean_features = np.mean(matrix, axis=1, keepdims=True)
    # vectorize operation to get covariance matrix - don't want to do an expensive python for loop
    num_examples = matrix.shape[1]
    return 1 / (num_examples - 1) * (matrix - mean_features).dot(
        (matrix - mean_features).T)
