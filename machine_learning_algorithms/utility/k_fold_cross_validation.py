""" This module contains code for a class representing the
k fold cross validation algorithm """
import copy
from typing import Union
import numpy as np
from machine_learning_algorithms.utility.misc import convertToHighestPred
from machine_learning_algorithms.neural_net_utility.optimizer import GradientDescent


class KFoldCrossValidation(object):
    """ This class represents the k fold cross validation
    algorithm """

    def get_k_score(self,
                    x: np.ndarray,
                    y: np.ndarray,
                    scoreFunc,
                    model,
                    k: int = 5,
                    numEpochs: Union[int, None] = None,
                    learn_rate: Union[float, None] = None,
                    optim=GradientDescent()):
        """ This function implements k-fold cross validation. This function
        should be used when your dataset is to small for a train/validation/test
        split to yield accurate estimates of generalizability.

        Args:
            x:
                Numpy array of shape (num_features, num_examples) containing
                feature vectors the algorithm is meant to train on

            y:
                Numpy array of shape (1, num_examples)

            scoreFunc:
                Function indicating how you would like to score your model
                Ex. Regression: MSE, MAE. Classification: Acccuracy,
                Precision, etc

            model:
                Model object that has a .fit() method and a .predict() method

            k:
                Integer representing how many folds total you would like to
                have in your estimation

            numEpochs:
                Value of type integer or None representing the number of epochs
                to train the algorithm, or None

            learn_rate:
                Floating point value representing the learning rate to use
                inside the optimization algorithm, or None

            optim:
                Object representing the optimization algorithm to use during
                training, or None

        Returns:
            Floating point value representing the average value of the score
            function over all the folds
        """
        batches = self.getBatches(x, y, k)
        # Loop over all k batches, and designate one batch to be the test set
        # and train on the other k-1 batches
        orig_object = model
        prediction_scores = []
        for i in range(k):
            hold_out_set = batches[i][0]
            # IE - one hotted labels need to be converted to a 1D array
            # of predictions
            if y.shape[0] > 1:
                hold_out_labels = convertToHighestPred(batches[i][1]).reshape(
                    1, -1)
            else:
                hold_out_labels = batches[i][1]
            # Discard the current model. Need to make a deepcopy to avoid
            # carrying around one set of learned weights from one iteration to another
            new_model = copy.deepcopy(orig_object)
            # For every set that is not the test set, include the labels and the examples and concatenate
            # it all at the end to get the overall train set
            xToTrain = [batches[i][0] for j in range(len(batches)) if j != i]
            yToTrain = [batches[i][1] for j in range(len(batches)) if j != i]
            curr_train = np.concatenate((xToTrain), axis=1)
            curr_labels = np.concatenate((yToTrain), axis=1)
            if not numEpochs and not learn_rate:
                new_model.fit(curr_train, curr_labels)
            else:
                new_model.fit(curr_train,
                              curr_labels,
                              num_epochs=numEpochs,
                              learn_rate=learn_rate,
                              optim=optim)

            # preds are shape (examples,). Incorrect to check the accuracy so we
            # reshape it to be (1, examples) which is what our labels are.
            preds = new_model.predict(hold_out_set).reshape(1, -1)
            assert hold_out_labels.shape == preds.shape, "hold out labels shape (%s, %s), preds shape is (%s, %s)" % (
                hold_out_labels.shape[0], hold_out_labels.shape[1],
                preds.shape[0], preds.shape[1])
            prediction_scores.append(scoreFunc(hold_out_labels, preds))
        return np.mean(prediction_scores)

    def getBatches(self, x, y, k):
        """
        This function randomly shuffles the data and then batches it into k
        batches with equal number of examples to use for k-fold CV. 

        Parameters: 
        - x (NumPy matrix) -> NumPy matrix of shape (features, examples) 

        - y (NumPy vector) -> NumPy vector of shape (1, examples)

        Returns: Output (list[tuples]) -> List containing tuples of examples. Ex.
        list[i] = (x_traini, y_traini)
        """
        # set a seed for random so you don't get different results every time you run the function
        ## extend capabilities of getBatches to one hot encoded target
        np.random.seed(seed=21)
        batches = []
        matrix_x = x.T
        y_exRow = y.T
        matrix = np.hstack((matrix_x, y_exRow))
        size_batch = x.shape[1] // k
        # np.random.shuffle only works for (M, N) matrices so after vertically stacking the two
        # matrices we tranpose and then shuffle
        np.random.shuffle(matrix)
        x_shuffled = matrix[:, :x.shape[0]].T
        if y.shape[0] == 1:
            y_shuffled = matrix[:, -1].T.reshape(1, -1)
        else:
            y_shuffled = matrix[:, x.shape[0]:].T
        batch_end = 0
        for i in range(k):
            batch_start = batch_end
            batch_end = batch_start + size_batch
            # Notation [:, batch_start: batch_start + batch_end] means to get all the rows for the current batch
            # of examples out of the matrix, since our matrix is of shape [features, examples] so we need to slice the
            # examples out
            batches.append((x_shuffled[:, batch_start:batch_end],
                            y_shuffled[:, batch_start:batch_end]))
        return batches
