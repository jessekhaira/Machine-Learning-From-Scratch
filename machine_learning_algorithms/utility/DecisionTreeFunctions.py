import numpy as np
from machine_learning_algorithms.utility.score_functions import total_sum_of_squares
from machine_learning_algorithms.utility.score_functions import entropy
from machine_learning_algorithms.utility.score_functions import giniIndex


def predictionClassification(labels):
    # Just predict the most commonly occurring class AKA the mode of the labels
    vals, counts = np.unique(labels, return_counts=True)
    idx = np.argmax(counts)
    return vals[idx]


def predictionRegression(labels):
    # Just predict the average of the values that fall in this leaf!
    return np.mean(labels)


def entropyGain(root, left, right):
    entropyRoot = entropy(root)
    entropyLeftNode = entropy(left)
    entropyRightNode = entropy(right)
    numExamplesLeft = left.shape[1]
    numExamplesRight = right.shape[1]
    fractionOfDataLeft, fractionOfDataRight = getFractions(
        numExamplesLeft, numExamplesRight)
    assert (
        fractionOfDataLeft + fractionOfDataRight
    ) == 1, "Somethings wrong with how your data is splitting into left and right datasets"
    # Intuitively, we want a feature that splits the data perfectly into pure nodes on the left and right side
    # meaning that going from the root node to the left nodes and right nodes, we gain a lot of information
    return entropyRoot - (fractionOfDataLeft * entropyLeftNode +
                          fractionOfDataRight * entropyRightNode)


def getFractions(numExamplesLeft, numExamplesRight):
    fracL = numExamplesLeft / (numExamplesLeft + numExamplesRight)
    fracR = 1 - fracL
    return fracL, fracR


def giniGain(root, left, right):
    giniCurr = giniIndex(root)
    giniL = giniIndex(left)
    giniR = giniIndex(right)
    numExamplesLeft = left.shape[1]
    numExamplesRight = right.shape[1]
    fracL, fracR = getFractions(numExamplesLeft, numExamplesRight)
    assert (
        fracL + fracR
    ) == 1, "Somethings  wrong with how your data is splitting into left and right datasets"
    return giniCurr - (fracL * giniL + fracR * giniR)


def varianceReduction(root, left, right):
    # In a regression tree, at any node, the expected value of all of the examples that fall in the node
    # IS the prediction. So getting the variance is like calculating the RSS, except our prediction for every
    # example is the same of the mean value
    varianceRoot = total_sum_of_squares(root)
    varianceLeft = total_sum_of_squares(left)
    varianceRight = total_sum_of_squares(right)
    numExamplesLeft = left.shape[1]
    numExamplesRight = right.shape[1]
    fracL, fracR = getFractions(numExamplesLeft, numExamplesRight)
    assert (
        fracL + fracR
    ) == 1, "Somethings  wrong with how your data is splitting into left and right datasets"
    # Ideally you have 0 variance in left node and 0 variance in right node since your predictions are just perfect! :D
    return varianceRoot - (fracL * varianceLeft + fracR * varianceRight)
