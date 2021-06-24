""" This module contains tests for the recurrent neural network defined
in the package """
from ML_algorithms.Supervised_Learning.Classifiers.recurrent_network import ReccurentNetLanguageModel
from ML_algorithms.Utility.misc import getUniqueChars, mapcharToIdx, mapidxToChar
from ML_algorithms.neural_net_utility.activation_functions import TanH
import unittest
import sys
import os


class TestRecurrentNetwork(unittest.TestCase):
    """ This class contains unit tests for the recurrent neural network defined
    in the package """

    def setUp(self):
        self.data = open(os.path.join(sys.path[0], "shake.txt"), "r").read()
        self.chars = getUniqueChars(self.data)
        self.idx_char_map = mapidxToChar(self.chars)
        self.char_idx_map = mapcharToIdx(self.chars)

    def testOverfitSmallBatch(self):
        # this makes sure the forward pass and the backward pass are
        # working correctly before doing expensive optimization
        rnn = ReccurentNetLanguageModel(self.idx_char_map,
                                        self.char_idx_map,
                                        TanH(),
                                        550,
                                        len(self.chars),
                                        temperature=0.3)
        rnn.fit(self.data[:100],
                25,
                verbose=True,
                num_epochs=5000,
                learn_rate=0.1)

    def testFitData(self):
        # takes a long time to fit but achieves okay performance
        rnn = ReccurentNetLanguageModel(self.idx_char_map,
                                        self.char_idx_map,
                                        TanH(),
                                        550,
                                        len(self.chars),
                                        temperature=0.3)
        rnn.fit(self.data, 25, verbose=True, num_epochs=5000, learn_rate=0.1)


if __name__ == "__main__":
    unittest.main()
