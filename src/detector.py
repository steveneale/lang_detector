# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'detector.py'

Detect the language of a given input text, using a given trained model.

2019 Steve Neale <steveneale3000@gmail.com>
"""

from src.io import ModelIO
from src import Vectoriser


class Detector(object):

    def __init__(self, gram_size: int = 2):
        """ Initialise the Detector class

        Keyword Arguments:
            gram_size {int} -- gram size for vectorising text input (default: {2})
        """
        self.grams = gram_size

    def detect_language(self, input_text: str, model_path: str):
        """ Detect the language of a given input text

        Arguments:
            input_text {str} -- input text
            model_path {str} -- path to a language detection model

        Returns:
            str -- language of the input text
        """
        model = ModelIO.load_model_from_path(model_path)
        X = Vectoriser(gram_size=self.grams).transform([input_text])
        y = model.predict(X)[0]
        return y
