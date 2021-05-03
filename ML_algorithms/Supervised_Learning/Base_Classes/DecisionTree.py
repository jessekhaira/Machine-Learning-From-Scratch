""" This module contains code representing the base class for CART
trees """
import numpy as np
from typing import Callable, Union, Tuple


class DecisionTreeNode(object):
    """ This class represents a node inside of a decision tree.

    Every node inside of a decision tree has three things:
        - A portion of data indicating the region of the data that
        fell into this node
        - A pointer to a left child
        - A pointer to a right child

    At inference time, we DFS the binary decision tree, compare our
    input vectors values based on the value of the feature + split
    point chosen at this node.

    N - number of features
    M - number of examples

    Attributes:
        x:
            A (N,M) numpy matrix containing feature vectors

        y:
            A (1,M) numpy vector containing label vectors for the feature
            vectors
    """

    def __init__(self):
        # Left and Right Branch connecting to this node
        self.left = None
        self.right = None
        # Feature + splitPt on feature used to split here if split
        self.feature_row = None
        self.split_pt_feature = None
        self.gain = None
        # If leaf node (no left child or right child), we will store the
        # prediction here
        self.prediction = None


class BaseDecisionTree(object):
    """ This class is a template for classification and regression trees (CART),
    from which you can instantiate classification trees for binary and
    multiclasss classification, and regression trees.

    Attributes:
        root:
            An object of type DecisionTreeNode representing the root of the
            decision tree

        trainingFunction:
            For classification, can either be giniGain or entropyGain,
            for regression should be variance reduction. This function
            will be used to construct the tree.

        predictionFunc:
            Function used to get predictions. For classification, will
            be the mode of the labels that lie in a given node. For
            regression, will be the average of the labels that lie in
            a given node.

        minSamplesSplit:
            Integer indicating the minimum number of examples that have
            to fall in this node to justify splitting further

        maxDepth:
            Integer representing the maximum depth to grow this tree

        maxFeatures:
            Integer representing the maximum number of features to use to
            determine the split

        min_impurity_decrease:
            Floating point value representing the minimum decrease in impurity
            to justify splitting a node and continuing training
    """

    def __init__(self,
                 trainingFunction: Callable[
                     [DecisionTreeNode, DecisionTreeNode, DecisionTreeNode],
                     float],
                 predictionFunc: Callable[[np.ndarray], float],
                 minSamplesSplit: int = 2,
                 maxDepth: int = None,
                 maxFeatures: int = None,
                 min_impurity_decrease: float = 0):

        self.root = DecisionTreeNode()
        self.trainFunc = trainingFunction
        self.predictionFunc = predictionFunc
        self.minSamplesSplit = minSamplesSplit
        self.maxDepth = maxDepth
        self.maxFeatures = maxFeatures
        self.min_impurity_decrease = min_impurity_decrease

    def fit(self, xtrain: np.ndarray, ytrain: np.ndarray) -> None:
        """ This method implements the .fit() method for decision trees, where
        we build the decision tree on the training set xtrain and ytrain.

        Decision trees train in a greedy manner as we are not globally
        optimizing the cost function. To do that, we would have to
        exhaustively build every single possible decision tree and
        evaluate each of them, and then choose the best out of that,
        which is very expensive to do.

        Here we simply just choose the best decision at each node and don't
        backtrack to change it.

        N - number of features
        M - number of examples

        Args:
            xtrain:
                A (N,M) numpy matrix consisting of feature vectors

            ytrain:
                A (1,M) numpy vector consisting of labels for the feature
                vectors
        """
        node = self.root
        self._recursiveTreeConstruction(node, xtrain, ytrain, 0)

    def _recursiveTreeConstruction(self, node: DecisionTreeNode,
                                   xtrain: np.ndarray, ytrain: np.ndarray,
                                   depth: int) -> None:
        """ This method recursively builds the tree out in a depth first manner.

        N - number of features
        M - number of examples

        Args:
            node:
                Object of type DecisionTreeNode, representing the current node
                we are on

            xtrain:
                A (N,M) numpy matrix representing the training vectors for the
                algorithm

            ytrain:
                A (1,M) numpy vector where M is the number of examples

            depth:
                Integer representing how deep in the tree the algorithm
                currently is
        """
        # Deal with base cases first - cases to stop building the tree out
        # If your depth is equal to the maximum depth allowed, just get the
        # prediction for this node and return
        if self.maxDepth is not None and depth == self.maxDepth:
            node.prediction = self.predictionFunc(ytrain)
            return

        # If the number of labels at this node is <= to the minimum number of
        # samples needed to justify a split then we set this node as a leaf
        # node and leave it with a prediction
        if ytrain.shape[1] <= self.minSamplesSplit:
            node.prediction = self.predictionFunc(ytrain)
            return

        # If there is only one unique label in this node, then we just predict
        # that label. No need to keep splitting further
        if len(np.unique(ytrain)) == 1:
            node.prediction = self.predictionFunc(ytrain)
            return

        # If we are doing feature bagging, then maxFeatures will not be None
        # and therefore we must slice out a random subsection of the features
        # and only use them to build the node

        # Sample without replacement!
        random_features_chosen = None
        if self.maxFeatures:
            random_features_chosen = np.random.choice(xtrain.shape[0],
                                                      self.maxFeatures,
                                                      replace=False)
            features = xtrain[random_features_chosen, :]
            assert features.shape == (
                self.maxFeatures, xtrain.shape[1]
            ), "Your sliced out features are shape (%s, %s) and xtrain is shape (%s, %s)" % (
                features.shape[0], features.shape[1], xtrain.shape[0],
                xtrain.shape[1])
        else:
            features = xtrain

        # Find the feature that best splits the labels and how well it does
        # at this task
        feature_row, split_pt, decrease_impurity = self._find_best_feature(
            features, ytrain, random_features_chosen)
        # if the decrease in impurity doesn't justify a split at this node,
        # then we set this node as a leaf node and return.
        if decrease_impurity <= self.min_impurity_decrease:
            node.prediction = self.predictionFunc(ytrain)
            return
        # Otherwise assign this node the feature col and split pt and
        # continue on
        node.feature_row = feature_row
        node.split_pt_feature = split_pt
        node.gain = decrease_impurity
        xtrainL, ytrainL, xtrainR, ytrainR = self._splitData(
            xtrain, ytrain, feature_row, split_pt)
        node.left = DecisionTreeNode()
        node.right = DecisionTreeNode()
        self._recursiveTreeConstruction(node.left, xtrainL, ytrainL, depth + 1)
        self._recursiveTreeConstruction(node.right, xtrainR, ytrainR, depth + 1)

    def _splitData(
            self, xtrain: np.ndarray, ytrain: np.ndarray, feature_row: int,
            split_pt: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ This method splits the feature vectors and labels into
        two portions based on the split pt. These are used to construct
        the left branch and right branch of the current node.

        N - number of features
        M - number of examples

        Args:
            xtrain:
                A (N,M) numpy matrix containing feature vectors

            ytrain:
                A (1,M) numpy vector containing the labels for xtrain

            feature_row:
                Integer value indicating the row the feature is on that we are
                splitting with

            split_pt:
                Integer value indicating the value we are splitting the feature
                at

        Returns:
            A tuple containg 4 elements.

            The first element is a numpy matrix containing the feature vectors
            for the left node and the second element is a numpy vector
            containing the labels for the feature vectors sent to the left node.

            The third element is a numpy matrix containing the feature vectors
            for the right node and the fourth element is a numpy vector
            containing the labels for the feature vectors sent to the right
            node.
        """
        xtrain = xtrain.T
        ytrain = ytrain.T
        # Stack them up horizontally so we can filter the matrix appropriately
        # IE so the labels stay attached to the correct feature vectors
        matrix = np.hstack((xtrain, ytrain))
        # Have to use isinstance here instead of checking type(x) is int or float
        # because we have np.float64 or np.int64 not base python ints
        # but those objects inherited from the base python ints so they will be instances of it
        if isinstance(split_pt,
                      (int, np.integer)) or isinstance(split_pt,
                                                       (float, np.float)):
            matrixTrue = matrix[matrix[:, feature_row] >= split_pt]
            xtrainR = matrixTrue[:, :-1].T
            ytrainR = matrixTrue[:, -1].T.reshape(1, -1)

            matrixFalse = matrix[matrix[:, feature_row] < split_pt]
            xtrainL = matrixFalse[:, :-1].T
            ytrainL = matrixFalse[:, -1].T.reshape(1, -1)
            return xtrainL, ytrainL, xtrainR, ytrainR
        else:
            matrixTrue = matrix[matrix[:, feature_row] == split_pt]
            xtrainR = matrixTrue[:, :-1].T
            ytrainR = matrixTrue[:, -1].T.reshape(1, -1)

            matrixFalse = matrix[matrix[:, feature_row] != split_pt]
            xtrainL = matrixFalse[:, :-1].T
            ytrainL = matrixFalse[:, -1].T.reshape(1, -1)

            return xtrainL, ytrainL, xtrainR, ytrainR

    def _find_best_feature(
            self,
            features: np.ndarray,
            ytrain: np.ndarray,
            features_chosen=None) -> Tuple[int, Union[int, str], int]:
        """ This method finds the feature + split pt pair that produces
        the overall highest gain at the current node in the decision tree.

        N - number of features
        M - number of examples

        Args:
            features:
                A (N,M) numpy matrix representing feature vectors

            ytrain:
                A (1,M) numpy vector representing labels for the feature vectors

        Returns:
            Tuple containing an integer representing the row of the feature that
            produced the highest gain, an integer or string representing the
            split pt of the feature that produced the highest gain (as we can
            split on continuous feature or discrete feature), and an integer
            representing the highest gain obtained at the node.
        """

        best_feature_produced_gain = -1
        best_split_pt_feature = -1
        highest_gain = -1
        # Loop over every single feature to try out all its split pts
        for feature_row in range(features.shape[0]):
            # extract the current feature out of the feature matrix, and
            # then find all of its unique values each unique value is a
            # possible split point
            curr_feature = features[feature_row, :].reshape(1, -1)
            possible_split_pts = np.unique(curr_feature)
            curr_gain = 0
            split_pt = -1
            for threshold_val in possible_split_pts:
                _, ytrain_l, _, ytrain_r = self._splitData(
                    features, ytrain, feature_row, threshold_val)
                # If the split produces zero examples for the left or
                # the right node, then we cannot consider this split
                # pt as valid
                if ytrain_l.shape[1] == 0 or ytrain_r.shape[1] == 0:
                    continue
                gain = self.trainFunc(ytrain, ytrain_l, ytrain_r)
                if gain > curr_gain:
                    curr_gain = gain
                    split_pt = threshold_val

            if curr_gain >= highest_gain:
                highest_gain = curr_gain
                best_feature_produced_gain = feature_row if features_chosen is None else features_chosen[
                    feature_row]
                best_split_pt_feature = split_pt

        return (best_feature_produced_gain, best_split_pt_feature, highest_gain)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """ This function implements the predict function for decision trees
        using the depth first search traversal algorithm.

        Args:
            x:
                A numpy vector or matrix that contains the same features as the
                decision tree was trained on

        Returns:
            A numpy vector containing the predictions of the decision tree on
            the input x
        """
        node = self.root
        # handle easy case when there's just a single vector to predict on
        if x.shape[1] == 1:
            return self._depth_first_search(x.reshape(-1, 1), node)
        node = self.root
        output = np.zeros((1, x.shape[1]))
        for i in range(x.shape[1]):
            output[0, i] = self._depth_first_search(x[:, i].reshape(-1, 1),
                                                    node)
        return output

    def _depth_first_search(self, feature_vector, node):
        if not node:
            raise "You reached a null node before reaching a prediction\
                ,there is an error somewhere"

        # We have reached a leaf node - get the prediction
        if not node.left and not node.right:
            return node.prediction
        # If its not a leaf node, we decide whether we want to traverse
        # down the left branch or right branch which depends on the
        # leafs feature +split point

        # As decision trees don't need categoric features to be encoded,
        # we will have different behaviour depending on the type of the
        # feature we used to split the tree at this node

        # being >= to feature val means you 'passed' test so you go to
        # right branch else you failed test and go to left branch
        if isinstance(node.split_pt_feature,
                      (int, np.integer)) or isinstance(node.split_pt_feature,
                                                       (float, np.float)):
            # feature vector will be a 1D column vector of shape (N,1)
            # so we just acccess the feature row and compare the value
            if feature_vector[node.feature_row] >= node.split_pt_feature:
                return self._depth_first_search(feature_vector, node.right)
            else:
                return self._depth_first_search(feature_vector, node.left)
        else:
            if feature_vector[node.feature_row] == node.split_pt_feature:
                return self._depth_first_search(feature_vector, node.right)
            else:
                return self._depth_first_search(feature_vector, node.left)

    def print_tree(self):
        """ This method recursively prints out the tree built, starting
        from the root node.
        """
        node = self.root
        self._print_helper(node)

    def _print_helper(self, node):
        if not node:
            return
        elif node.prediction is not None:
            print(
                "We've arrived at a leaf node and prediction! Prediction: %s" %
                (node.prediction))
            print("\n")
        else:
            print("Comparison test: feature at row: %s at split pt: %s" %
                  (node.feature_row, node.split_pt_feature))
            print("\n")
            print("If test passed, going to right branch")
            self._print_helper(node.right)
            print("\n")
            print("If test failed, going to left branch")
            self._print_helper(node.left)
