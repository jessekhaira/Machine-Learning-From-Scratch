""" This module contains tests for the recurrent neural network defined
in the package """
from ML_algorithms.Supervised_Learning.Classifiers.recurrent_network import ReccurentNetLanguageModel
from ML_algorithms.Utility.misc import getUniqueChars, mapcharToIdx, mapidxToChar
from ML_algorithms.Neural_Net_Util.ActivationFunctions import TanH
import unittest
import sys
import os

data = open(os.path.join(sys.path[0], "shake.txt"), "r").read()
chars = getUniqueChars(data)
idxCharMap = mapidxToChar(chars)
charIdxMap = mapcharToIdx(chars)


class TestRecurrentNetwork(unittest.TestCase):
    """ This class contains unit tests for the recurrent neural network defined
    in the package """

    def testOverfitSmallBatch(self):
        # this makes sure the forward pass and the backward pass are
        # working correctly before doing expensive optimization
        RNN_model = ReccurentNetLanguageModel(idxCharMap,
                                              charIdxMap,
                                              TanH(),
                                              550,
                                              len(chars),
                                              temperature=0.3)
        RNN_model.fit(data[:100],
                      25,
                      verbose=True,
                      num_epochs=5000,
                      learn_rate=0.1)

    def testFitData(self):
        # takes a long time to fit but achieves okay performance
        RNN_model = ReccurentNetLanguageModel(idxCharMap,
                                              charIdxMap,
                                              TanH(),
                                              550,
                                              len(chars),
                                              temperature=0.3)
        RNN_model.fit(data, 25, verbose=True, num_epochs=5000, learn_rate=0.1)


if __name__ == "__main__":
    unittest.main()
