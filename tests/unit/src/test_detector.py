# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'test_detector.py'

Unit tests for 'detector.py'

2018-2019 Steve Neale <steveneale3000@gmail.com>
"""

import unittest

from mock import patch, Mock

from src import Detector


class TestDetector(unittest.TestCase):

    def setUp(self):
        self.detector = Detector(gram_size=3)

    def test_correctly_initialised_with_gram_size(self):
        self.assertEqual(self.detector.grams, 3)

    @patch("src.detector.Vectoriser")
    @patch("src.io.ModelIO.load_model_from_path")
    def test_detect_language(self, mock_load_model, mock_vectoriser):
        # Mock objects, classes and methods
        mocked_model = Mock()
        mock_load_model.return_value = mocked_model
        mocked_vectoriser = mock_vectoriser.return_value
        mocked_vectoriser.transform.return_value = [1, 2, 3, 4]
        mocked_model.predict.return_value = [1, 2]
        # Run the 'detect_language' method
        detected = self.detector.detect_language("This is a test", "path/to/model")
        # Make assertions
        mock_load_model.assert_called_with("path/to/model")
        mock_vectoriser.assert_called_with(gram_size=3)
        mocked_vectoriser.transform.assert_called_with(["This is a test"])
        mocked_model.predict.assert_called_with([1, 2, 3, 4])
        self.assertEqual(detected, 1)

    def tearDown(self):
        del self.detector


if __name__ == "__main__":
    unittest.main()
