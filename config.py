#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
'config.py'

Configuration file for 'lang_detector'.

2017 Steve Neale <steveneale3000@gmail.com>
"""

config = { # The default list of supported languages
           "languages": { "de": "German",
                          "en": "English",
                          "es": "Spanish",
                          "fr": "French",
                          "it": "Italian",
                          "pt": "Portuguese"
                        },
           # The default dataset to use for model training
           "dataset": "./data/default_data.csv",
           # The default training and test set sizes for model training/evaluation
           "train_size": 45000,
           "test_size": 5000,
           # N-gram size for model training
           "grams": 2,
           # The default trained model to use for detection
           "default_model": "./models/default_model"
         }