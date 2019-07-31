#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'detector.py'

Detect the language of a given input text, using a given trained model.

2019 Steve Neale <steveneale3000@gmail.com>
"""

from src.io import ModelIO
from src import Vectoriser


class Detector:

    def __init__(self, gram_size=2):
        self.grams = gram_size

    def detect_language(self, input_text, model_path):
        model = ModelIO.load_model_from_path(model_path)
        X = Vectoriser(gram_size=self.grams).transform([input_text])
        y = model.predict(X)[0]
        return y
