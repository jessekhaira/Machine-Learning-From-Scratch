import numpy as np
class kNearestNeighbours_base(object):
    """
    This is a minimal implementation of the nonparametric supervised
    machine learning algorithm called kNN, used for classification. 

    Params:

    k -> Number of neighbours to consider when making a prediction. 
    Default to consider is 10.

    similarity_metric -> Metric to use to determine the similarity between 
    different vectors. Default is euclidean distance (L2 distance).
    """
    def __init__(self, k=10, similarity_metric="L2", verbose = True):
        #Allow either L2 distance or L1 distance to be used
        assert similarity_metric == "L2" or similarity_metric == "L1", "Sorry, you must use either L2 distance or L1 distance"
        self.k = k
        self.similarity_metric = similarity_metric
        self.verbosity = verbose
    
    def fit(self, x_train, y_train): 
        """
        This method trains the object on the feature vectors (x_train),
        and the vectors corresponding labels (y_train).

        Input:
        x_train (NumPy matrix)-> NumPy matrix of shape (M,Nx) where Nx is the number
        of features in one example, and M is the number of training examples.

        y_train (NumPy vector)-> NumPy vector of shape (M,1) where M is the number of 
        training examples.
        
        Output: 
        None
        """
        # kNN just memorizes the data - the model we are "learning" is the data.
        # no need to train parameters. This is called lazy learning. 
        assert x_train.shape[0] == y_train.shape[0]
        self.modelX = x_train
        self.modelY = y_train

    def predict(self, x_predict):
        """
        This method takes in a matrix of unlabelled vectors, and uses the
        trained kNN model to get predictions for all the vectors.

        Input:
        x_predict (NumPy matrix) -> NumPy matrix of shape (M,Nx) where Nx
        is the number of features in one example, and M is the number of 
        examples.

        Output(NumPy vector) -> Column vector of shape (M,1) containing the 
        labels for all the M examples.
        """
        assert x_predict.shape[1] == self.modelX.shape[1], "The input should have the same number of features as what you trained on!"
        y_pred = np.empty((x_predict.shape[0], 1))
        # Loop over every example in x_predict, get k_closest neighbours
        # and assign the prediction for this example
        for i,example in enumerate(x_predict):
            print('Predicting on the %sth example, %s/%s'%(i, i, len(x_predict)))
            k_closest = self._getKClosest(example, self.similarity_metric)
            y_pred[i] = self._getPrediction(k_closest)
        return y_pred

    def _getKClosest(self, ex_x, similarity):
        """
        This method returns the k closest vectors to the current vector ex_x 
        in the training set. 

        Input:
        ex_x (NumPy vector) -> (1,N) vector
        similarity -> "L2" or "L1" indicating the type of similarity metric being used

        Output (NumPy matrix) -> Matrix of shape (K, 1) containing the K closest vectors
        with their corresponding labels 
        """
        similarity = 2 if similarity == "L2" else 1 
        distance_to_all = np.linalg.norm(self.modelX-ex_x, ord=similarity, axis=1, keepdims = True)
        assert distance_to_all.shape == (self.modelX.shape[0], 1), "You should have a distance vector of shape (m,1)!"
        return self.modelY[np.argpartition(distance_to_all, self.k, axis=0)][:self.k]

    def _getPrediction(self, k_closest):
        """
        This method is the only place where the kNN regressor and classifier differ.
        Classifier -> take the most common class among the neighbours.
        Regressor -> take the average among the neighbours. 

        Thus, this method will be left as an abstract method, to be inherited by 
        both the kNN classifier and regressor and overriden with their own specific implementations.
        """
        raise NotImplementedError 