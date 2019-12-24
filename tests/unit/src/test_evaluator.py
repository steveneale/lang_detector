# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'test_evaluator.py'

Unit tests for 'evaluator.py'

2018-2019 Steve Neale <steveneale3000@gmail.com>
"""

import unittest

import numpy as np
import pandas as pd

from numpy import ndarray

from mock import patch

from src import Evaluator


class TestEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = Evaluator(gram_size=4)
        self.test_dataframe = pd.DataFrame(data={"sentence": ["This is a test", np.nan],
                                                 "language": ["en", "fr"]})
        self.sentence_values = pd.Series(["This is a test", ""]).values

    def test_correctly_initialised_with_gram_size(self):
        self.assertEqual(self.evaluator.grams, 4)

    @patch("src.evaluator.Vectoriser")
    @patch("src.evaluator.pd.Series.fillna")
    def test_get_test_sentences_and_labels(self, mock_fillna, mock_vectoriser):
        # Mock classes and methods
        mock_fillna.return_value = ["This is a test", ""]
        mocked_vectoriser = mock_vectoriser.return_value
        mocked_vectoriser.transform.return_value = [[1, 2, 3, 4], []]
        # Run the 'detect_language' method
        X, y = self.evaluator._get_test_sentences_and_labels(self.test_dataframe)
        # Make assertions
        mock_fillna.assert_called_with("")
        mock_vectoriser.assert_called_with(gram_size=4)
        mocked_vectoriser.transform.assert_called
        self.assertIsInstance(X, list)
        self.assertEqual(X, [[1, 2, 3, 4], []])
        self.assertIsInstance(y, ndarray)
        self.assertEqual(list(y), ["en", "fr"])

    def tearDown(self):
        del self.evaluator
        del self.test_dataframe
