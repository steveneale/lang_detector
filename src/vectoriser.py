#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
'vectoriser.py'

Hashing vectoriser class

2018-2019 Steve Neale <steveneale3000@gmail.com>
"""

import re

from sklearn.feature_extraction.text import HashingVectorizer

class Vectoriser():

    def __init__(self, tokenisation="gramify", gram_size=2):

        tokeniser = getattr(self, tokenisation)
        self.gram_size = gram_size
        self.vectoriser = HashingVectorizer(decode_error="ignore",
                                            alternate_sign=False,
                                            n_features=2**21,
                                            preprocessor=None,
                                            tokenizer=self.tokeniser)


    def gramify(text):

        grams = []
        if text == None or text == "":
            return grams
        text = "_{}_".format(re.sub(r"[^\w\s']", "", text).strip().replace(" ", "_").lower())
        for i in range(0, len(text)-1, 1):
            if self.gram_size == 2:
                grams.append("{}{}".format(text[i], text[i+1]))
            elif self.gram_size == 3 and i < len(text)-2:
                grams.append("{}{}{}".format(text[i], text[i+1], text[i+2]))
        return grams


