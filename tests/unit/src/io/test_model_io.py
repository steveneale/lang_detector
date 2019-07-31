#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'test_model_io.py'

Unit tests for 'model_io.py' (src.io)

2019 Steve Neale <steveneale3000@gmail.com>
"""

import unittest

from mock import patch

from src.io import ModelIO


class TestModelIO(unittest.TestCase):

    @patch("src.io.model_io.load_pickled_model")
    @patch("src.io.model_io.os")
    def test_load_model_from_path_if_path_exists(self, mock_os, mock_load_pickled):
        mock_os.path.exists.return_value = True
        mock_load_pickled.return_value = object
        model = ModelIO.load_model_from_path("models/test/path")
        self.assertEqual(model, object)
        mock_load_pickled.assert_called_with("models/test/path")

    @patch("src.io.model_io.os")
    def test_load_model_from_path_if_path_not_exists(self, mock_os):
        mock_os.path.exists.return_value = False
        with self.assertRaises(FileNotFoundError) as context:
            model = ModelIO.load_model_from_path("models/test/path")
            self.assertEqual(model, None)
        self.assertEqual("The given model path could not be found.", str(context.exception))

    @patch("src.io.model_io.save_pickled_model")
    def test_save_model_to_destination(self, mock_save_pickled):
        ModelIO.save_model_to_destination(object, "test/destination")
        mock_save_pickled.assert_called_with(object, "test/destination")
