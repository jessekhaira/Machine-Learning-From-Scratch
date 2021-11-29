""" This module contains tests for the recurrent neural network defined
in the package """
import unittest
import sys
import os
from machine_learning_algorithms.supervised_learning.classifiers.recurrent_network import ReccurentNetLanguageModel
from machine_learning_algorithms.utility.misc import get_unique_chars, map_char_to_idx, map_idx_to_char
from machine_learning_algorithms.neural_net_utility.activation_functions import TanH


class TestRecurrentNetwork(unittest.TestCase):
    """ This class contains unit tests for the recurrent neural network defined
    in the package """

    def setUp(self):
        self.data = open(os.path.join(sys.path[0], "shake.txt"), "r").read()
        self.chars = get_unique_chars(self.data)
        self.idx_char_map = map_idx_to_char(self.chars)
        self.char_idx_map = map_char_to_idx(self.chars)

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
