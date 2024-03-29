""" This module contains code representing the unsupervised
machine learning algorithm principal component analysis """
import numpy as np
from machine_learning_algorithms.utility.misc import get_covariance_matrix


class PrincipalComponentAnalysis:
    """ Principal Component Analysis (PCA) is a unsupervised machine
    learning algorithm primarily used for dimensionality reduction
    and exploratory data analysis.

    PCA can either be carried out by eigendecomposition on the
    covariance matrix created from the design matrix, or by
    directly applying Singular Value Decomposition to the design
    matrix itself.

    The result is finding dimensions that are uncorrelated with
    each other that have the maximum variance.
    """

    def __init__(self):
        self.eigen_values = None
        self.eigen_vectors = None

    def fit_transform(self, data: np.ndarray, num_principal: int) -> np.ndarray:
        """ This method extracts num_principal principal components
        from the input data and returns them.

        Args:
            data:
                NumPy array of shape (num_features, num_examples) representing
                the data to find principal components for
            num_principal:
                Integer representing the number of dimensions that the user
                would like to reduce the data to
        Returns:
            NumPy array of shape (num_principal, num_examples) where
            num_principal is the number of features in the reduced
            dimensionality dataset.
        """
        # perform PCA by doing eigen decomposition on the covariance matrix
        cov_matrix = get_covariance_matrix(data)
        # we lose interpretability of the features but we get uncorrelated
        # features

        # eigenvalues - how much variance is captured by a certain dimension
        # eigenvectors - the new axes (dimensions) we will use
        self.eigen_values, self.eigen_vectors = np.linalg.eig(cov_matrix)
        # order the eigenvectors according to the eigenvalues - IE the
        # eigenvalue matrix is really like a diagonal covariance matrix
        # in that every element is 0 (meaning no dimension is correlated
        # with any other) except for the diagonal elements, which indicate
        # how much variance the i == j dimension explains
        idxSorted = np.argsort(self.eigen_values, axis=0)[::-1]
        eigen_vectors = self.eigen_vectors[idxSorted]
        # transform the data to the new vector space with the principal
        # components found above and only return the first num_principal
        # dimensions
        return self.eigen_vectors.dot(data)[:num_principal]
