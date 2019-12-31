# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'data_processor.py'

Process data for use with language detection models (training and/or evaluation).

2019 Steve Neale <steveneale3000@gmail.com>
"""

import pandas as pd

from typing import List

from sklearn.model_selection import train_test_split

from src.exceptions import CSVLoadingException


class DataProcessor(object):

    def __init__(self):
        pass

    def get_train_and_test_from_csv(self, csv_path: str, test_size: int = 0.25, seed: int = None):
        data = self._load_dataframe_from_csv(csv_path)
        train_data, test_data = train_test_split(data["sentence"].values,
                                                 data["language"].values,
                                                 test_size=test_size,
                                                 shuffle=True,
                                                 stratify=data["language"].values,
                                                 random_state=seed)
        return train_data, test_data

    @staticmethod
    def _load_dataframe_from_csv(csv_path: str, headers: List = None):
        try:
            data = pd.read_csv(csv_path)
            if headers is not None and not all([x for x in headers if x in list(data.columns)]):
                raise ValueError("Some of the required headers ({}) were not found " +
                                 "in the given CSV file ({})".format(headers, csv_path))
            return data
        except Exception as err:
            raise CSVLoadingException("A problem was encountered while trying to " +
                                      "load a DataFrame from CSV: {}".format(err))
