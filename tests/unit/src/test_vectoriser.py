#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
'test_vectoriser.py'

Unit tests for 'vectoriser.py'

2018-2019 Steve Neale <steveneale3000@gmail.com>
"""

import unittest

from src import Vectoriser


class TestVectoriser(unittest.TestCase):

    def test_gramify(self):

        self.vectoriser = Vectoriser(tokenisation="gramify")
        grams = self.vectoriser.gramify("This, is the 1st sentence.")
        self.assertEqual(grams, ["_t", "th", "hi", "is", "s_", "_i", "is", "s_", 
                                 "_t", "th", "he", "e_", "_1", "1s", "st", "t_", 
                                 "_s", "se", "en", "nt", "te", "en", "nc", "ce", "e_"])



if __name__ == "__main__":
    unittest.main()