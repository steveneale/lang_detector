# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'evaluator.py'

Evaluate a given trained language detection model, using given test data.

2019 Steve Neale <steveneale3000@gmail.com>
"""

import pandas as pd

from pandas import DataFrame

from src.io import ModelIO
from src import Vectoriser

from typing import Any, Tuple, List


class Evaluator(object):

    def __init__(self, gram_size: int = 2):
        """ Initialise the Evaluator class

        Keyword Arguments:
            gram_size {int} -- gram size for vectorising text input (default: {2})
        """
        self.grams = gram_size

    def evaluate_model(self, model_path: str, test_path: str, for_languages: str = "all") -> Any:
        """ Evaluate a given language detection model

        Arguments:
            model_path {str} -- path to a language detection model
            test_path {str} -- path to test data

        Keyword Arguments:
            for_languages {str} -- whether to evaluate 'all' or 'each' language(s) (default: {"all"})

        Raises:
            ValueError: 'for_languages' option must be either 'all' or 'each'

        Returns:
            [Any] -- accuracy, or a dictionary containing the accuracy for each language
        """
        if for_languages not in ["all", "each"]:
            raise ValueError("'for_languages' option must be either 'all' or 'each'")
        model = ModelIO.load_model_from_path(model_path)
        test_data = pd.read_csv(test_path)
        return self._get_model_accuracy(model, test_data, for_languages=for_languages)

    def _get_model_accuracy(self, model: str, test_data: str, for_languages: str = "all") -> Any:
        """ Get the accuracy of a given model

        Arguments:
            model {str} -- path to a language detection model
            test_data {str} -- path to test data

        Keyword Arguments:
            for_languages {str} -- whether to evaluate 'all' or 'each' language(s) (default: {"all"})

        Returns:
            Any -- accuracy, or a dictionary containing the accuracy for each language
        """
        if for_languages == "all":
            X, y = self._get_test_sentences_and_labels(test_data)
            return model.score(X, y)
        elif for_languages == "each":
            language_accuracies = {}
            for language in test_data["language"].unique():
                language_data = test_data.iloc[test_data["language"] == language]
                X, y = self._get_test_sentences_and_labels(language_data)
                language_accuracies[language] = model.score(X, y)
            return language_accuracies

    def _get_test_sentences_and_labels(self, test_data: DataFrame) -> Tuple[List[str], List[str]]:
        """ Get sentences and (language) labels from a Pandas DataFrame

        Arguments:
            test_data {DataFrame} -- Pandas DataFrame containing test data

        Returns:
            Tuple[List[str], List[str]] -- lists of test sentences and corresponding language labels
        """
        test_data["sentence"] = test_data["sentence"].fillna("")
        X = Vectoriser(gram_size=self.grams).transform(test_data["sentence"].values)
        y = test_data["language"].values
        return X, y
