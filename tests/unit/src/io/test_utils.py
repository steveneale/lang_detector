#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'test_utils.py'

Unit tests for 'utils.py' (src.io)

2018-2019 Steve Neale <steveneale3000@gmail.com>
"""

import unittest
from mock import patch

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


if __name__ == "__main__":
    unittest.main()
