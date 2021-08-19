import unittest
import numpy as np
from sklearn.datasets import load_iris
from sklearn import preprocessing
from machine_learning_algorithms.unsupervised_learning.k_means import KMeansClustering
import matplotlib.pyplot as plt

X, Y = load_iris(return_X_y=True)

X = preprocessing.scale(X).T


class tests(unittest.TestCase):

    def testIris(self):
        kMeanObj = KMeansClustering()
        clustersAssigned = kMeanObj.fit_predict(X, num_clusters=3)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # as expected, we got the wrong clusters which is what we should've gotten :D
        scatter = []
        markers = ['.', ',', 'o']
        colors = ['red', 'blue', 'green']
        for i, m, color in zip(range(3), markers, colors):
            ithClusterPts = np.where(clustersAssigned[:] == i)[1]
            scatterI = ax.scatter(X[0, ithClusterPts],
                                  X[1, ithClusterPts],
                                  X[3, ithClusterPts],
                                  c=color,
                                  marker=m)
            scatter.append(scatterI)
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
