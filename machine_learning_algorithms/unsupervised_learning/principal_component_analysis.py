import numpy as np
from machine_learning_algorithms.utility.misc import getCovarianceMatrix


class PrincipalComponentAnalysis:
    """
    Principal Component Analysis (PCA) is a dimensionality reduction algorithm primarily used for exploratory
    data analysis and for making predictive models. PCA can either be carried out by eigendecomposition
    on the covariance matrix created from the design matrix, or by directly applying Singular Value Decomposition to the
    design matrix itself. 

    The result is finding dimensions that are uncorrelated with each other that have the maximum variance. 

    Parameters:
    -> data (NumPy Matrix): NumPy matrix with features on the rows and examples on the columns. 
    -> num_principal (int): The number of dimensions that the user would like to reduce the data to 

    Returns:
    -> data_transformed (NumPy Matrix): NumPy matrix of shape (num_principal, M) where num_principal is the number of 
    features in the reduced dimensionality dataset, and M is the number of examples. 
    """

    def __init__(self):
        self.eigenValues = None
        self.eigenVectors = None

    def fit_transform(self, data, num_principal):
        # perform PCA by doing eigen decomposition on the covariance matrix
        cov_matrix = getCovarianceMatrix(data)
        # we lose interpretability of the features but we get uncorrelated features

        # eigenvalues - how much variance is captured by a certain dimension
        # eigenvectors - the new axes (dimensions) we will use
        self.eigenValues, self.eigenVectors = np.linalg.eig(cov_matrix)

        # order the eigenvectors according to the eigenvalues - IE the eigenvalue matrix is really
        # like a diagonal covariance matrix in that every element is 0 (meaning no dimension is correlated with any other)
        # except for the diagonal elements, which indicate how much variance the i == j dimension explains
        idxSorted = np.argsort(self.eigenValues, axis=0)[::-1]
        eigenVectors = self.eigenVectors[idxSorted]
        # transform the data to the new vector space with the principal components found above
        # and only return the first num_principal dimensions
        return self.eigenVectors.dot(data)[:num_principal]
