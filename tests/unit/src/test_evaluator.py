# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'test_evaluator.py'

Unit tests for 'evaluator.py'

2018-2019 Steve Neale <steveneale3000@gmail.com>
"""

import unittest

from mock import patch, Mock

from src import Evaluator


class TestEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = Evaluator(gram_size=4)

    def test_correctly_initialised_with_gram_size(self):
        self.assertEqual(self.evaluator.grams, 4)