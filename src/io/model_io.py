#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'model_io.py'

Model-specific input/output functions.

2019 Steve Neale <steveneale3000@gmail.com>
"""

import os

from src.io.utils import load_pickled_model, save_pickled_model


class ModelIO:

    @staticmethod
    def load_model_from_path(model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError("The given model path could not be found.")
        model = load_pickled_model(model_path)
        return model

    @staticmethod
    def save_model_to_destination(model, destination):
        save_pickled_model(model, destination)
