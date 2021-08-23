import numpy as np
from machine_learning_algorithms.utility.misc import euclideanDistance
import copy


class KMeansClustering(object):
    """
    This class represents the K-Means algorithm, which is an
    unsupervised machine learning algorithm used to find hidden structure
    inside of data.

    Attributes:
        maxNumLoops:
            Integer representing the maximum number of times the algorithm
            should run (if it does not converge on its own).

        clusterCentroids:
            Value that is None or is a numpy array holding vectors in the input
            data that are considered cluster centroids

        clustersVectorsAssigned:
            Value that is None or is a numpy array indicating the cluster
            centroid every vector in the input data will have
    """

    def __init__(self, maxNumLoops=300):
        # need a container to hold each cluster centroid
        self.clusterCentroids = None
        # need a container to hold the vectors each cluster has been assigned to
        self.clustersVectorsAssigned = None
        self.maxNumLoops = maxNumLoops

    def fit_predict(self,
                    data,
                    num_clusters,
                    similarityMetric=euclideanDistance,
                    seed=21):
        """ This method computes cluster centroids for every vector in the data
        and returns them.

        Args:
            data:
                Numpy array of shape (num_features, num_examples) indicating the
                data to find clusters in

            num_clusters:
                Integer representing the number of clusters the user would
                like to discover in the data

            similarityMetric:
                Function used to compute the similarity between vectors

            seed:  
                Integer representing the seed to use when randomly generating
                the cluster centroids

        Returns:
            A numpy array of shape (1, num_examples) indicating the cluster
            every vector in the data has been assigned to.
        """

        self._initializeClusterCentroids(num_clusters, data, seed)
        self.clustersVectorsAssigned = np.zeros((1, data.shape[1]))

        # we assign vectors to the cluster centroid they are the closest to
        # and then update the cluster centroid until convergence
        for _ in range(self.maxNumLoops):
            for i in range(data.shape[1]):
                vector = data[:, i]
                self._updateCluster(vector, similarityMetric, i)
            # update cluster centroids to be the avg of the (new) points assigned to it
            oldClusterCentroids = copy.copy(self.clusterCentroids)
            self._updateClusterCentroids(data)
            # CONVERGENCE check - did cluster centroids change at all
            didCentroidsChange = oldClusterCentroids - self.clusterCentroids
            # if all values are zero, then that means no cluster centroids changed
            # and we're done
            if not didCentroidsChange.any():
                break
        return self.clustersVectorsAssigned

    def _updateClusterCentroids(self, data):
        for i in range(self.clusterCentroids.shape[1]):
            # get all pts assigned to this cluster
            pts_cluster = np.where(self.clustersVectorsAssigned[:] == i)[1]
            # pull out the vectors from the data
            matrix_cluster = data[:, pts_cluster]
            # change the cluster centroid to be the average of the pts assigned to this cluster
            self.clusterCentroids[:, i] = np.mean(matrix_cluster, axis=1)

    def _updateCluster(self, vector, similarityMetric, vectorPos):
        # this method updates the cluster centroid each vector is assigned to
        minDistance = float('inf')
        cluster = -1
        # compare this vector to every single cluster centroid and assign this vector
        # to the cluster centroid it is the closest to (most similar to)
        for i in range(self.clusterCentroids.shape[1]):
            center = self.clusterCentroids[:, i]
            distance_ithCentroid = similarityMetric(vector, center)
            if distance_ithCentroid < minDistance:
                minDistance = distance_ithCentroid
                cluster = i

        assert cluster != -1, "The cluster this vector is assigned to is still -1, somethings wrong"
        self.clustersVectorsAssigned[:, vectorPos] = cluster

    def _initializeClusterCentroids(self, num_clusters, data, seed):
        # this method randomly chooses vectors in the data to be cluster centroids
        np.random.seed(seed)
        self.clusterCentroids = np.zeros((data.shape[0], num_clusters))
        for i in range(num_clusters):
            idx_vector = np.random.choice(data.shape[1])
            self.clusterCentroids[:, i] = data[:, idx_vector]
