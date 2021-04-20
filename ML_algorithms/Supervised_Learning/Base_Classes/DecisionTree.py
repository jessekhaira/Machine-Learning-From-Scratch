""" This module contains code representing a generic decision tree """
import numpy as np
import sys
from typing import Callable


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
        self.splitPtFeature = None
        self.gain = None
        # If leaf node (no left child or right child), we will store the prediction here
        self.prediction = None


class BaseDecisionTree(object):
    """ This class is a template for classification and regression trees (CART),
    from which you can instantiate classification trees for binary and
    multiclasss classification, and regression trees.

    Attributes:
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

    def fit(self, xtrain, ytrain):
        """
        This method implements the .fit() method for decision trees, where we build the decision tree on the training set:
        xtrain and ytrain. 

        Decision trees train in a greedy manner as we are not globally optimizing the cost function. To do that, we would have to exhaustively
        build every single possible decision tree and evaluate each of them, and then choose the best out of that, which is an NP-Complete problem.

        Here we simply just choose the best decision at each node and don't backtrack to change it. 

        Parameters:
        -> xtrain (NumPy matrix): A (N,M) matrix where N is features and M is examples 
        -> ytrain (NumPy vector): A (1,M) vector where M is the number of examples 

        Returns: None 
        """
        node = self.root
        self._recursiveTreeConstruction(node, xtrain, ytrain, 0)

    def _recursiveTreeConstruction(self, node, xtrain, ytrain, depth):
        """
        This method recursively builds the tree out, building each branch out in a depth first manner.

        Parameters:
        -> node (DecisionTreeNode): Current node in the binary decision tree
        -> xtrain (NumPy matrix): A (N,M) matrix where N is features and M is examples 
        -> ytrain (NumPy vector): A (1,M) vector where M is the number of examples 
        -> depth (int): Integer representing how deep in the tree the algorithm currently is 

        Returns: None 
        """
        # Deal with base cases first - cases to stop building the tree out
        # If your depth is equal to the maximum depth allowed, just get the prediction for this node and return
        if self.maxDepth is not None and depth == self.maxDepth:
            node.prediction = self.predictionFunc(ytrain)
            return

        # If the number of labels at this node is <= to the minimum number of samples needed to justify a split
        # then we set this node as a leaf node and leave it with a prediction
        if ytrain.shape[1] <= self.minSamplesSplit:
            node.prediction = self.predictionFunc(ytrain)
            return

        # If there is only one unique label in this node, then we just predict that label. No need to keep splitting further
        if len(np.unique(ytrain)) == 1:
            node.prediction = self.predictionFunc(ytrain)
            return

        # If we are doing feature bagging, then maxFeatures will not be None and therefore we must slice out
        # a random subsection of the features and only use them to build the node

        # Sample without replacement!
        randomFeaturesChosen = None
        if self.maxFeatures:
            randomFeaturesChosen = np.random.choice(xtrain.shape[0],
                                                    self.maxFeatures,
                                                    replace=False)
            features = xtrain[randomFeaturesChosen, :]
            assert features.shape == (
                self.maxFeatures, xtrain.shape[1]
            ), "Your sliced out features are shape (%s, %s) and xtrain is shape (%s, %s)" % (
                features.shape[0], features.shape[1], xtrain.shape[0],
                xtrain.shape[1])
        else:
            features = xtrain

        # Find the feature that best splits the labels and how well it does at this task
        feature_row, splitPt, decreaseImpurity = self._findBestFeature(
            features, ytrain, randomFeaturesChosen)
        # if the decrease in impurity doesn't justify a split at this node, then we set this node as a leaf
        # node and return.
        if decreaseImpurity <= self.min_impurity_decrease:
            node.prediction = self.predictionFunc(ytrain)
            return
        # Otherwise assign this node the feature col and split pt and continue on
        node.feature_row = feature_row
        node.splitPtFeature = splitPt
        node.gain = decreaseImpurity
        xtrainL, ytrainL, xtrainR, ytrainR = self._splitData(
            xtrain, ytrain, feature_row, splitPt)
        node.left = DecisionTreeNode()
        node.right = DecisionTreeNode()
        self._recursiveTreeConstruction(node.left, xtrainL, ytrainL, depth + 1)
        self._recursiveTreeConstruction(node.right, xtrainR, ytrainR, depth + 1)

    def _splitData(self, xtrain, ytrain, feature_row, splitPt):
        """
        This method splits the feature vectors and labels into two portions based
        on the split pt. These are used to construct the left branch and right branch of the
        current node.

        Parameters:
        -> xtrain (NumPy matrix): A (N,M) matrix where N is features and M is examples 
        -> ytrain (NumPy vector): A (1,M) vector where M is the number of examples 
        -> feature_row (int): Integer value indicating the row the feature is on that we are splitting with
        -> splitPt (int): Integer value indicating the value we are splitting the feature at

        Returns:
        -> xtrainL (NumPy matrix): A (N,M) matrix where N is features and M is examples where the feature_row
        is < than (if numeric) or != the splitPt (if categoric)

        -> ytrainL (NumPy vector): A (1,M) vector where M is the number of examples where the feature_row
        is < than (if numeric) or != the splitPt (if categoric)

        -> xtrainR (NumPy matrix): A (N,M) matrix where N is features and M is examples where the feature_row is
        >= than (if numeric) or == splitPt (if categoric)

        -> ytrainR (NumPy vector):  A (1,M) vector where M is the number of examples where the feature_row is
        >= than (if numeric) or == splitPt (if categoric) 
        """
        xtrain = xtrain.T
        ytrain = ytrain.T
        # Stack them up horizontally so we can filter the matrix appropriately
        # IE so the labels stay attached to the correct feature vectors
        matrix = np.hstack((xtrain, ytrain))
        # Have to use isinstance here instead of checking type(x) is int or float
        # because we have np.float64 or np.int64 not base python ints
        # but those objects inherited from the base python ints so they will be instances of it
        if isinstance(splitPt,
                      (int, np.integer)) or isinstance(splitPt,
                                                       (float, np.float)):
            matrixTrue = matrix[matrix[:, feature_row] >= splitPt]
            xtrainR = matrixTrue[:, :-1].T
            ytrainR = matrixTrue[:, -1].T.reshape(1, -1)

            matrixFalse = matrix[matrix[:, feature_row] < splitPt]
            xtrainL = matrixFalse[:, :-1].T
            ytrainL = matrixFalse[:, -1].T.reshape(1, -1)
            return xtrainL, ytrainL, xtrainR, ytrainR
        else:
            matrixTrue = matrix[matrix[:, feature_row] == splitPt]
            xtrainR = matrixTrue[:, :-1].T
            ytrainR = matrixTrue[:, -1].T.reshape(1, -1)

            matrixFalse = matrix[matrix[:, feature_row] != splitPt]
            xtrainL = matrixFalse[:, :-1].T
            ytrainL = matrixFalse[:, -1].T.reshape(1, -1)

            return xtrainL, ytrainL, xtrainR, ytrainR

    def _findBestFeature(self, features, ytrain, featuresChosen=None):
        """
        This method finds the feature + split pt pair that produces the overall highest gain at the current node in the
        decision tree. 

        Parameters:
        -> features (NumPy matrix): A (N,M) matrix where N is features and M is examples 
        -> ytrain (NumPy vector): A (1,M) vector where M is the number of examples 

        Returns:
        -> bestFeature_ProducedGain (int): Row of the feature that produced the highest gain 
        -> bestSplit_ptFeature (int or str, depends on the type of the feature): Split pt of the feature that produced the highest gain 
        -> highestGain (int): Integer representing the highest gain obtained at the node. HAS to be higher than min_impurity_decrease, otherwise
        we will return -1's and make the current node a leaf node. 
        """

        bestFeature_ProducedGain = -1
        bestSplit_ptFeature = -1
        highestGain = -1
        # Loop over every single feature to try out all its split pts
        for feature_row in range(features.shape[0]):
            # extract the current feature out of the feature matrix, and then find all of its unique values
            # each unique value is a possible split point
            currFeature = features[feature_row, :].reshape(1, -1)
            possibleSplitPts = np.unique(currFeature)
            currGain = 0
            split_pt = -1
            for thresholdVal in possibleSplitPts:
                xtrainL, ytrainL, xtrainR, ytrainR = self._splitData(
                    features, ytrain, feature_row, thresholdVal)
                # If the split produces zero examples for the left or the right node, then we cannot consider this split
                # pt as valid
                if ytrainL.shape[1] == 0 or ytrainR.shape[1] == 0:
                    continue
                gain = self.trainFunc(ytrain, ytrainL, ytrainR)
                if gain > currGain:
                    currGain = gain
                    split_pt = thresholdVal

            if currGain >= highestGain:
                highestGain = currGain
                bestFeature_ProducedGain = feature_row if featuresChosen is None else featuresChosen[
                    feature_row]
                bestSplit_ptFeature = split_pt

        return (bestFeature_ProducedGain, bestSplit_ptFeature, highestGain)

    def predict(self, x):
        """
        This function implements the predict function for decision trees. 

        Parameters:
        -> x (vector or matrix): A single vector, or a matrix of vectors that need to be assigned labels.

        Returns: Output(NumPy vector) containing the predictions from the tree for every single example. 
        """
        node = self.root
        # handle easy case when there's just a single vector to predict on
        if x.shape[1] == 1:
            return self._DFS(x.reshape(-1, 1), node)
        node = self.root
        output = np.zeros((1, x.shape[1]))
        for i in range(x.shape[1]):
            output[0, i] = self._DFS(x[:, i].reshape(-1, 1), node)
        return output

    def _DFS(self, feature_vector, node):
        if not node:
            raise (
                "You reached a null node before reaching a prediction, there is an error somewhere"
            )
        # We have reached a leaf node - get the prediction
        if not node.left and not node.right:
            return node.prediction
        # If its not a leaf node, we decide whether we want to traverse down the left branch or right branch
        # which depends on the leafs feature +split point

        # As decision trees don't need categoric features to be encoded, we will have different behaviour depending on the
        # type of the feature we used to split the tree at this node

        # being >= to feature val means you 'passed' test so you go to right branch
        # else you failed test and go to left branch
        if isinstance(node.splitPtFeature,
                      (int, np.integer)) or isinstance(node.splitPtFeature,
                                                       (float, np.float)):
            # feature vector will be a 1D column vector of shape (N,1) so we just acccess the feature row and compare the value
            if feature_vector[node.feature_row] >= node.splitPtFeature:
                return self._DFS(feature_vector, node.right)
            else:
                return self._DFS(feature_vector, node.left)
        else:
            if feature_vector[node.feature_row] == node.splitPtFeature:
                return self._DFS(feature_vector, node.right)
            else:
                return self._DFS(feature_vector, node.left)

    def printTree(self):
        """
        This method recursively prints out the tree built, starting from the root node.

        Input: None
        Returns: None 
        """
        node = self.root
        self._printHelper(node)

    def _printHelper(self, node):
        if not node:
            return
        elif node.prediction is not None:
            print(
                "We've arrived at a leaf node and prediction! Prediction: %s" %
                (node.prediction))
            print('\n')
        else:
            print("Comparison test: feature at row: %s at split pt: %s" %
                  (node.feature_row, node.splitPtFeature))
            print('\n')
            print("If test passed, going to right branch")
            self._printHelper(node.right)
            print('\n')
            print("If test failed, going to left branch")
            self._printHelper(node.left)
