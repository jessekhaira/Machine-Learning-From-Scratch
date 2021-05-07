""" This module contains code representing the general
bagged forest supervised machine learning algorithm """
import numpy as np
from ML_algorithms.Supervised_Learning.Regression.RegressionTree import RegressionTree
from ML_algorithms.Supervised_Learning.Classifiers.classificationTree import ClassificationTree
from ML_algorithms.Utility.DecisionTreeFunctions import predictionClassification
from ML_algorithms.Utility.DecisionTreeFunctions import predictionRegression
from ML_algorithms.Utility.ScoreFunctions import MSE, RMSE, accuracy
from typing import Literal, Union


class BaggedForest(object):
    """
    This class represents bootstrap aggregated (bagged) decision trees.

    Attributes:
        typeSupervised:
            Integer that can be two values: 0 or 1. If 0, train classification
            models. If 1, train regression models.

        criterion:
            String representing the type of loss function to use. If
            classification, input should either be "gini" or "entropy". If
            regression, can be None.

        num_estimators:
            Integer representing the number of estimators to include in the
            ensemble

        max_samples:
            None, integer, or floating point value representing the number of
            samples to put into each bootstrap sample. If None, the num sample
            is equal to the dataset size. If int, the number of samples is
            equal to the size of the int. If float, the float should be between
            0 and 1 and it will indicate the percentage to include in each
            bootstrap sample.

        bootstrap:
            Boolean indicating whether to bootstrap the dataset when building
            the ensemble.

        minSamplesSplit:
            Integer indicating the minimum number of examples that have to fall
            in this node to justify splitting further

        maxDepth:
            Integer representing the maximum depth to grow this tree

        maxFeatures:
            Integer representing the maximum number of features to use to
            determine the split

        min_impurity_decrease:
            Integer representing the minimum decrease in impurity to justify
            splitting the node

        verbose:
            Boolean indicating whether to give updates when fitting and
            predicting
    """

    def __init__(self,
                 typeSupervised: Literal[0, 1],
                 criterion: Literal["gini", "entropy", None],
                 num_estimators: int = 100,
                 max_samples: Union[None, int, float] = None,
                 bootstrap: bool = False,
                 minSamplesSplit: int = 2,
                 maxDepth: int = None,
                 maxFeatures: int = None,
                 min_impurity_decrease: int = 0,
                 verbose: bool = False):
        self.typeSupervised = typeSupervised
        self.criterion = criterion
        self.num_estimators = num_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.minSamplesSplit = minSamplesSplit
        self.maxDepth = maxDepth
        self.maxFeatures = maxFeatures
        self.min_impurity_decrease = min_impurity_decrease
        self.verbose = verbose
        # variable that keeps track of the examples used to train
        # every tree in the ensemble so we can get OOB accuracy and error
        self.examplesUsedInBootstrap = []

        # variable indicating if this bagged forest is a random forest
        # ensemble holder
        self.forest = self._build_forest()

    def _build_forest(self):
        """ This method builds the forest of trees out using the specifications
        given in the constructor.

        The individual trees will be trained when the .fit() method is called.
        """
        forest = []
        # If classification, fit classification trees with the appropriate
        # criterion otherwise, fit regression trees
        if self.typeSupervised == 0:
            criterion = True if self.criterion == "entropy" else False
            for i in range(self.num_estimators):
                forest.append(
                    ClassificationTree(
                        entropy=criterion,
                        minSamplesSplit=self.minSamplesSplit,
                        maxDepth=self.maxDepth,
                        maxFeatures=self.maxFeatures,
                        min_impurity_decrease=self.min_impurity_decrease))
            return forest
        else:
            for i in range(self.num_estimators):
                forest.append(
                    RegressionTree(
                        minSamplesSplit=self.minSamplesSplit,
                        maxDepth=self.maxDepth,
                        maxFeatures=self.maxFeatures,
                        min_impurity_decrease=self.min_impurity_decrease))
            return forest

    def fit(self, xtrain: np.ndarray, ytrain: np.ndarray) -> None:
        """ This method implements the .fit() method for bagged forests,
        where we build the forest on xtrain and ytrain.

        N - number of features
        M - number of examples

        Args:
            xtrain:
                A (N,M) matrix containing feature vectors for the ensemble
                to train on

            ytrain:
                A (1,M) vector containing labels for the feature vectors
        """
        # Determine the size of each sample with which to train every tree
        if isinstance(self.max_samples, int):
            boot_strap_size = self.max_samples
        elif isinstance(self.max_samples, float):
            boot_strap_size = int(self.max_samples * xtrain.shape[1])
        else:
            boot_strap_size = xtrain.shape[1]

        for i in range(len(self.forest)):
            # If we are bootstrapping, then we need to generate a bootstrap
            # sample, record the examples used to train the tree, and fit
            # the tree on the boostrapped samples with the boostrapSize examples

            # If we aren't boostrapping, just fit every tree on the entire
            # training set
            if self.bootstrap:
                # sample with replacement when we decide which examples to use
                # to train the ith tree
                examples_in_boot_strapped_sample = np.random.choice(
                    xtrain.shape[1], size=boot_strap_size, replace=True)
                x_boot_strapped = xtrain[:, examples_in_boot_strapped_sample]
                y_boot_strapped = ytrain[:, examples_in_boot_strapped_sample]
                self.examplesUsedInBootstrap.append(
                    set(examples_in_boot_strapped_sample))
                self.forest[i].fit(x_boot_strapped, y_boot_strapped)
            else:
                self.forest[i].fit(xtrain, ytrain)

            if self.verbose:
                print(f"Finished fitting the {i}st tree in the ensemble,\
                    {i}/{len(self.forest)} of the way there!")
                print("\n")

    def predict(self, x: np.ndarray):
        """ This method implements the .predict() method for bagged forests.
        In the .predict() method, we simply aggregate the predictions
        from each individual tree in the forest.

        N - number of features
        M - number of examples

        Args:
            x:
                A (N,M) matrix containing feature vectors to predict on 
        """

        predictions = np.zeros((1, x.shape[1]))
        for i in range(x.shape[1]):
            feature_vector = x[:, i].reshape(-1, 1)
            feature_vectorPreds = np.zeros((1, x.shape[1]))
            for j in range(len(self.forest)):
                predictionIthTree = self.forest[i].predict(feature_vector)
                feature_vectorPreds[:, i] = predictionIthTree
            if self.typeSupervised == 0:
                singlePrediction = predictionClassification(feature_vectorPreds)
            else:
                singlePrediction = predictionRegression(feature_vectorPreds)
            predictions[:, i] = singlePrediction
            if self.verbose:
                print(
                    "Finished predicting on the %sst example, %s/%s of the way there!"
                    % (i, i, len(self.forest)))
                print('\n')
        return predictions

    def getOOBScore(self, xtrain, ytrain):
        """
        This method gets the out of bag score for the model. The method loops over every single example 
        in the training set, and for every example, if the current tree did NOT fit on it, then the tree will
        predict on it. The predictions for every tree will be amalgamated into one prediction. 

        For classification, we will return the accuracy and the error on the out of bag sample.
        For regression, we will return the mean squared error(mse) and the root mean squared error (rmse)
        on the out of bag sample.

        Parameters:
        -> xtrain (NumPy matrix): A (N,M) matrix where N is features and M is examples 
        -> ytrain (NumPy vector): A (1,M) vector where M is the number of examples 

        Returns:
        -> If self.typeSupervised == 0, accuracy (int) and error (int)
        -> If self.typeSupervised == 1, mse (int) and rmse (int)
        """
        predictions = np.zeros((1, xtrain.shape[1]))
        for i in range(xtrain.shape[1]):
            feature_vector = xtrain[:, i].reshape(-1, 1)
            preds_currVector = []
            for j in range(len(self.forest)):
                # if this tree trained on this example, then skip it
                if i in self.examplesUsedInBootstrap[j]:
                    continue
                # otherwise predict on it
                prediction = self.forest[j].predict(feature_vector)
                preds_currVector.append(prediction)
            if self.typeSupervised == 0:
                overallPrediction = predictionClassification(preds_currVector)
            else:
                overallPrediction = predictionRegression(preds_currVector)
            predictions[:, i] = overallPrediction
        if self.typeSupervised == 0:
            acc = accuracy(ytrain, predictions)
            error = 1 - acc
            return acc, error
        else:
            mse = MSE(ytrain, predictions)
            rmse = RMSE(ytrain, predictions)
            return mse, rmse
