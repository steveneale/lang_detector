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

from mock import patch, Mock, call

from src import Evaluator


class TestEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = Evaluator(gram_size=4)
        self.test_dataframe = pd.DataFrame(data={"sentence": ["This is a test", ""],
                                                 "language": ["en", "fr"]})
        self.sentence_values = pd.Series(["This is a test", ""]).values

    def test_correctly_initialised_with_gram_size(self):
        self.assertEqual(self.evaluator.grams, 4)

    @patch("src.Evaluator._get_test_sentences_and_labels")
    def test_get_model_accuracy_for_all_languages(self, mock_sents_and_labels):
        # Mock objects, classes and methods
        mock_sents_and_labels.return_value = [[1, 2, 3, 4]], ["en"]
        mocked_model = Mock()
        mocked_model.score.return_value = 95.0
        # Run the '_get_model_accuracy' method
        accuracy = self.evaluator._get_model_accuracy(mocked_model,
                                                      self.test_dataframe,
                                                      for_languages="all")
        # Make assertions
        mock_sents_and_labels.assert_called_with(self.test_dataframe)
        mocked_model.score.assert_called_with([[1, 2, 3, 4]], ["en"])
        self.assertIsInstance(accuracy, float)
        self.assertEqual(accuracy, 95.0)

    @patch("src.Evaluator._get_test_sentences_and_labels")
    @patch("src.evaluator.pd.Series.unique")
    def test_get_model_accuracy_with_each_language(self, mock_unique, mock_sents_and_labels):
        # Mock objects, classes and methods
        mock_unique.return_value = ["en", "fr"]
        mock_sents_and_labels.side_effect = [
            ([[1, 2, 3, 4]], ["en"]),
            ([[5]], ["fr"])
        ]
        mocked_model = Mock()
        mocked_model.score.side_effect = [95.0, 97.0]
        # Run the '_get_model_accuracy' method
        accuracies = self.evaluator._get_model_accuracy(mocked_model,
                                                        self.test_dataframe,
                                                        for_languages="each")
        # Make assertions
        mock_unique.assert_called()
        self.assertEqual(mock_sents_and_labels.call_count, 2)
        self.assertEqual(mocked_model.score.call_count, 2)
        self.assertIsInstance(accuracies, dict)
        self.assertEqual(accuracies["en"], 95.0)
        self.assertEqual(accuracies["fr"], 97.0)

    @patch("src.evaluator.Vectoriser")
    @patch("src.evaluator.pd.Series.fillna")
    def test_get_test_sentences_and_labels(self, mock_fillna, mock_vectoriser):
        # Mock classes and methods
        mock_fillna.return_value = ["This is a test", ""]
        mocked_vectoriser = mock_vectoriser.return_value
        mocked_vectoriser.transform.return_value = [[1, 2, 3, 4], []]
        # Run the '_get_test_sentences_and_labels' method
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
