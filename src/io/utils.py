#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'utils.py' (src.io)

Input/output utility functions

2018-2019 Steve Neale <steveneale3000@gmail.com>
"""

import os
import pickle


def create_directory(path):
    destination = os.path.join(".", path)
    if not os.path.exists(destination):
        os.makedirs(destination)
    return destination


def save_pickled_model(model, destination):
    destination_file = open(os.path.join(destination, "pkl_objects", "model.pkl"), "wb")
    pickle.dump(model, destination_file, protocol=4)


def load_pickled_model(source_location):
    source_file = open(os.path.join(source_location, "pkl_objects", "model.pkl"), "rb")
    model = pickle.load(source_file)
    return model
