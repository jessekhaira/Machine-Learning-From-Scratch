import numpy as np
import unittest
from machine_learning_algorithms.neural_net_utility.loss_functions import cross_entropy


class testLossFunc(unittest.TestCase):

    def testCrossEntropy(self):
        y = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]).T
        yhat = np.array([[0.2, 0.4, 0.2, 0.2], [0.5, 0.2, 0.2, 0.1],
                         [0.4, 0.1, 0.2, 0.9]]).T

        CE = cross_entropy()

        loss = CE.get_loss(y, yhat)

        print(loss)

        derivLoss = CE.derivativeLoss_wrtPrediction(y, yhat)

        print(derivLoss)

        print(CE._gradient_checking(y, yhat))

        y1 = np.array([0, 0, 1, 0]).T
        yhat1 = np.array([0.32574286, 0.081362, 0.0352241, 0.55767104]).T
        print(CE.get_loss(y1, yhat1))


if __name__ == "__main__":
    unittest.main()