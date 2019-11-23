# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'test_detector.py'

Unit tests for 'detector.py'

2018-2019 Steve Neale <steveneale3000@gmail.com>
"""

import unittest

from mock import patch

from src import Detector


class TestDetector(unittest.TestCase):

    def setUp(self):
        self.detector = Detector(gram_size=3)

    def test_correctly_initialised_with_gram_size(self):
        self.assertEqual(self.detector.grams, 3)

    def tearDown(self):
        del self.detector


if __name__ == "__main__":
    unittest.main()
