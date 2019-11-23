# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'test_vectoriser.py'

Unit tests for 'vectoriser.py'

2018-2019 Steve Neale <steveneale3000@gmail.com>
"""

import unittest

from scipy.sparse import csr_matrix

from src import Vectoriser


class TestVectoriser(unittest.TestCase):

    def test_gramify_2_grams(self):
        vectoriser = Vectoriser(tokenisation="gramify")
        grams = vectoriser.gramify("This, is the 1st sentence.")
        self.assertEqual(grams, ["_t", "th", "hi", "is", "s_", "_i", "is", "s_", 
                                 "_t", "th", "he", "e_", "_1", "1s", "st", "t_", 
                                 "_s", "se", "en", "nt", "te", "en", "nc", "ce", "e_"])

    def test_gramify_3_grams(self):
        vectoriser = Vectoriser(tokenisation="gramify", gram_size=3)
        grams = vectoriser.gramify("This, is the 1st sentence.")
        self.assertEqual(grams, ["_th", "thi", "his", "is_", "s_i", "_is", "is_",
                                 "s_t", "_th", "the", "he_", "e_1", "_1s", "1st",
                                 "st_", "t_s", "_se", "sen", "ent", "nte", "ten",
                                 "enc", "nce", "ce_"])

    def test_gramify_empty_string(self):
        vectoriser = Vectoriser(tokenisation="gramify")
        grams = vectoriser.gramify("")
        self.assertEqual(grams, [])

    def test_transform(self):
        vectoriser = Vectoriser(tokenisation="gramify")
        vectorised = vectoriser.transform(["This, is the 1st sentence.",
                                           "Here is another sentence.",
                                           "This is the third sentence.",
                                           "The fourth and final sentence."])
        self.assertEqual(type(vectorised), csr_matrix)
        self.assertEqual(vectorised.shape[0], 4)


if __name__ == "__main__":
    unittest.main()
