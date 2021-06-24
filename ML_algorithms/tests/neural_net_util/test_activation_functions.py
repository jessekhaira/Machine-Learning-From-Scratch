import numpy as np
import unittest
from ML_algorithms.neural_net_utility.activation_functions import Softmax
from ML_algorithms.neural_net_utility.LossFunctions import cross_entropy
from ML_algorithms.neural_net_utility.neural_net_layers import dLdZ_sm


class testActivFuncs(unittest.TestCase):

    def test1(self):
        #test with single example first
        np.random.seed(21)
        sm = Softmax()
        ce = cross_entropy()
        Z1 = np.random.randn(4, 5)
        A1 = sm.compute_output(Z1)
        print(A1)
        Y = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0],
                      [0, 1, 0, 0]]).T
        print(ce.get_loss(Y, A1))
        print(Y)
        dLdA1 = ce.derivativeLoss_wrtPrediction(Y, A1)
        print(dLdA1)
        dLdZ = dLdZ_sm(Z1, A1, dLdA1, sm)
        # Only one value in each column should be negative, the rest should be positive
        # and the values should be easily correlated with (a-1)/m (if y =1) else (a)/m
        print(dLdZ)


if __name__ == "__main__":
    unittest.main()