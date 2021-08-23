""" This module contains unit tests for the k-means
clustering algorithm """
import unittest
import numpy as np
from sklearn.datasets import load_iris
from sklearn import preprocessing
from machine_learning_algorithms.unsupervised_learning.k_means import KMeansClustering
import matplotlib.pyplot as plt


class KMeansClusteringTests(unittest.TestCase):
    """ This class contains unit tests for the k-means
    clustering algorithm """

    def setUp(self):
        self.x, self.y = load_iris(return_X_y=True)
        self.x = preprocessing.scale(self.x).T

    def tearDown(self):
        self.x = None
        self.y = None

    def test1(self):
        k_mean_obj = KMeansClustering()
        clusters_assigned = k_mean_obj.fit_predict(self.x, num_clusters=3)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = []
        markers = ['.', ',', 'o']
        colors = ['red', 'blue', 'green']
        for i, m, color in zip(range(3), markers, colors):
            ith_cluster_pts = np.where(clusters_assigned[:] == i)[1]
            scatter_i = ax.scatter(self.x[0, ith_cluster_pts],
                                   self.x[1, ith_cluster_pts],
                                   self.x[3, ith_cluster_pts],
                                   c=color,
                                   marker=m)
            scatter.append(scatter_i)
        ax.legend(scatter, (
            'cluster1',
            'cluster2',
            'cluster3',
        ))
        ax.set_xlabel('Dim1')
        ax.set_ylabel('Dim2')
        ax.set_zlabel('Dim3')
        plt.show()


if __name__ == "__main__":
    unittest.main()
