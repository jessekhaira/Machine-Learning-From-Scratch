import sys
import os 
sys.path.append("/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Supervised_Learning/Classifiers")
sys.path.append("/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Neural_Net_Util")
sys.path.append("/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Utility")
sys.path.append("/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Supervised_Learning/Classifiers/Tests")
from RecurrentNet_languageModel import ReccurentNet_languageModelChar
from misc import getUniqueChars
from misc import mapidxToChar
from misc import mapcharToIdx
from ActivationFunctions import ReLU
from ActivationFunctions import TanH
from ActivationFunctions import Sigmoid
from Optimizers import gradientDescentMomentum
from Optimizers import Adam
from Optimizers import RMSProp 
from Optimizers import AdaGrad
import unittest
import numpy as np 


data = open('/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Supervised_Learning/Classifiers/Tests/shake.txt', 'r').read()
chars = getUniqueChars(data)
idxCharMap = mapidxToChar(chars)
charIdxMap = mapcharToIdx(chars)


class tests(unittest.TestCase):
    def testOverfitSmallBatch(self):
        # this makes sure the forward pass and the backward pass are working correctly before doing expensive optimization 
        RNN_model = ReccurentNet_languageModelChar(idxCharMap, charIdxMap, TanH(), 550, len(chars), temperature=0.3)
        RNN_model.fit(data[:100], 25, verbose=True, num_epochs=  5000, learn_rate=0.1)

    def testFitData(self):
        # takes a very long time to fit but achieves okay performance
        RNN_model = ReccurentNet_languageModelChar(idxCharMap, charIdxMap, TanH(), 550, len(chars), temperature=0.3)
        RNN_model.fit(data, 25, verbose=True, num_epochs=  5000, learn_rate=0.1)


    


if __name__ == "__main__":
    unittest.main()