""" This module contains code representing the unsupervised
machine learning algorithm K-Means """
import numpy as np
from machine_learning_algorithms.utility.misc import euclideanDistance
import copy


class KMeansClustering:
    """ This class represents the K-Means algorithm. The K-Means algorithm
    is an unsupervised machine learning algorithm used to find hidden
    structure inside of data.

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

    def __init__(self, maxNumLoops: int = 300):
        # need a container to hold each cluster centroid
        self.cluster_centroids = None
        # need a container to hold the vectors each cluster has been assigned to
        self.cluster_vectors_assigned = None
        self.max_num_loops = maxNumLoops

    def fit_predict(self,
                    data: np.ndarray,
                    num_clusters: int,
                    similarity_metric=euclideanDistance,
                    seed: int = 21):
        """ This method computes cluster centroids for every vector in the data
        and returns them.

        Args:
            data:
                Numpy array of shape (num_features, num_examples) indicating the
                data to find clusters in

            num_clusters:
                Integer representing the number of clusters the user would
                like to discover in the data

            similarity_metric:
                Function used to compute the similarity between vectors

            seed:
                Integer representing the seed to use when randomly generating
                the cluster centroids

        Returns:
            A numpy array of shape (1, num_examples) indicating the cluster
            every vector in the data has been assigned to.
        """

        self._init_cluster_centroids(num_clusters, data, seed)
        self.cluster_vectors_assigned = np.zeros((1, data.shape[1]))

        # we assign vectors to the cluster centroid they are the closest to
        # and then update the cluster centroid until convergence
        for _ in range(self.max_num_loops):
            for i in range(data.shape[1]):
                vector = data[:, i]
                self._update_cluster(vector, similarity_metric, i)
            # update cluster centroids to be the avg of the (new)
            # points assigned to it
            old_cluster_centroids = copy.copy(self.cluster_centroids)
            self._update_cluster_centroids(data)
            # CONVERGENCE check - did cluster centroids change at all
            did_centroids_change = (old_cluster_centroids -
                                    self.cluster_centroids)
            # if all values are zero, then that means no cluster
            # centroids changed and we're done
            if not did_centroids_change.any():
                break
        return self.cluster_vectors_assigned

    def _update_cluster_centroids(self, data: np.ndarray):
        for i in range(self.cluster_centroids.shape[1]):
            # get all pts assigned to this cluster
            pts_cluster = np.where(self.cluster_vectors_assigned[:] == i)[1]
            # pull out the vectors from the data
            matrix_cluster = data[:, pts_cluster]
            # change the cluster centroid to be the average of the pts
            # assigned to this cluster
            self.cluster_centroids[:, i] = np.mean(matrix_cluster, axis=1)

    def _update_cluster(self, vector: np.ndarray, similarity_metric,
                        vector_pos: int) -> None:
        # this method updates the cluster centroid each vector is assigned to
        min_distance = float("inf")
        cluster = -1
        # compare this vector to every single cluster centroid and
        # assign this vector to the cluster centroid it is the
        # closest to (most similar to)
        for i in range(self.cluster_centroids.shape[1]):
            center = self.cluster_centroids[:, i]
            dist_ith_centroid = similarity_metric(vector, center)
            if dist_ith_centroid < min_distance:
                min_distance = dist_ith_centroid
                cluster = i

        assert cluster != -1, ("The cluster this vector is assigned to " +
                               "is still -1, somethings wrong")
        self.cluster_vectors_assigned[:, vector_pos] = cluster

    def _init_cluster_centroids(self, num_clusters: int, data: np.ndarray,
                                seed: int) -> None:
        # this method randomly chooses vectors in the data to be
        # cluster centroids
        np.random.seed(seed)
        self.cluster_centroids = np.zeros((data.shape[0], num_clusters))
        for i in range(num_clusters):
            idx_vector = np.random.choice(data.shape[1])
            self.cluster_centroids[:, i] = data[:, idx_vector]
