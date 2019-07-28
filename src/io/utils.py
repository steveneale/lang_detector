#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'utils.py' (src.io)

Input/output utility functions

2018-2019 Steve Neale <steveneale3000@gmail.com>
"""

import os


def create_directory(path):
    destination = os.path.join(".", path)
    if not os.path.exists(destination):
        os.makedirs(destination)
    return destination
