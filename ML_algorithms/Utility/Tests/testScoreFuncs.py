import sys
sys.path.append("/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Supervised_Learning/Classifiers")
sys.path.append("/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Utility")
import unittest
import numpy as np 
from DecisionTreeFunctions import entropy
from DecisionTreeFunctions import entropyGain
from DecisionTreeFunctions import TSS
from DecisionTreeFunctions import varianceReduction 
from DecisionTreeFunctions import giniIndex

class tests(unittest.TestCase):
    """
    This class contains unit tests for each of the methods
    defined for the kNN class.
    """
    def testEntropy1(self):
        a = np.array([0, 3, 0, 1, 0]).reshape(1,-1)
        self.assertAlmostEqual(entropy(a), 0.9502705392)
    
    def testEntropyandInfoGain(self):
        root = np.array([0, 3, 0, 1, 0]).reshape(1,-1)
        left = np.array([0,0,0]).reshape(1,-1)
        right = np.array([1,3]).reshape(1,-1)
        entropyRoot = entropy(root)
        entropyL = entropy(left)
        entropyR = entropy(right)
        self.assertAlmostEqual(entropyRoot, 0.9502705392)
        self.assertEqual(entropyL, 0)
        self.assertAlmostEqual(entropyR, 0.6931471805)
        self.assertAlmostEqual(entropyGain(root, left, right), 0.6730116670092)

    def testVarianceReduction(self):
        root = np.array([0.32, 0.72, 0.51, 0.63, 0.92, 0.82, 0.4]).reshape(1, -1)
        left = np.array([0.32, 0.72, 0.51]).reshape(1, -1)
        right = np.array([0.63, 0.92, 0.82, 0.4]).reshape(1,-1)
        self.assertAlmostEqual(varianceReduction(root, left, right), 0.166242857)

    def testGini(self):
        root = np.array([0, 3, 0, 1, 0]).reshape(1,-1)
        self.assertAlmostEqual(giniIndex(root), 0.56)


if __name__ == "__main__":
    unittest.main()