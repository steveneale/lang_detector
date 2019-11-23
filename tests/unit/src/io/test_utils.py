# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'test_utils.py'

Unit tests for 'utils.py' (src.io)

2018-2019 Steve Neale <steveneale3000@gmail.com>
"""

import unittest
from mock import patch, MagicMock
from io import StringIO, BytesIO

from src.io import utils


class TestUtils(unittest.TestCase):

    @patch("src.io.utils.os")
    def test_create_directory_if_exists(self, mock_os):
        mock_os.path.join.return_value = "./output/test/directory"
        mock_os.path.exists.return_value = True
        directory = utils.create_directory("output/test/directory")
        self.assertEqual(directory, "./output/test/directory")

    @patch("src.io.utils.os")
    def test_create_directory_if_not_exists(self, mock_os):
        mock_os.path.join.return_value = "./output/test/directory"
        mock_os.path.exists.return_value = False
        directory = utils.create_directory("output/test/directory")
        self.assertEqual(directory, "./output/test/directory")

    @patch("src.io.utils.pickle")
    @patch("src.io.utils.os")
    def test_save_pickled_model_if_pkl_dir_exists(self, mock_os, mock_pickle):
        mock_os.path.exists.return_value = True
        mock_os.path.join.return_value = "output/test/pkl_objects/model.pkl"
        with patch("src.io.utils.open") as mock_open:
            mock_open.return_value = StringIO()
            utils.save_pickled_model(object, "output")
            mock_os.makedirs.assert_not_called()
            mock_pickle.dump.assert_called()

    @patch("src.io.utils.pickle")
    @patch("src.io.utils.os")
    def test_save_pickled_model_if_pkl_dir_not_exists(self, mock_os, mock_pickle):
        mock_os.path.exists.return_value = False
        mock_os.path.join.return_value = "output/test/pkl_objects/model.pkl"
        with patch("src.io.utils.open") as mock_open:
            mock_open.return_value = StringIO()
            utils.save_pickled_model(object, "output")
            mock_os.makedirs.assert_called()
            mock_pickle.dump.assert_called()

    @patch("src.io.utils.pickle")
    @patch("src.io.utils.os")
    def test_load_pickled_model(self, mock_os, mock_pickle):
        mock_os.path.join.return_value = "test/models/pkl_objects/model.pkl"
        with patch("src.io.utils.open") as mock_open:
            mock_open.return_value = BytesIO(b"...")
            model = utils.load_pickled_model("test/models")
            self.assertIsInstance(model, MagicMock)
            mock_pickle.load.assert_called()


if __name__ == "__main__":
    unittest.main()
