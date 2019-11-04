# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'lang_detector.py'

Train and evaluate language detection models, and predict input text language using them.

2019 Steve Neale <steveneale3000@gmail.com>
"""

import sys
import os
import re
import random
import datetime

import argparse
import pyprind

import pandas as pd
import numpy as np

from sklearn.naive_bayes import MultinomialNB

from config import config

from src.io.utils import create_directory, save_pickled_model, load_pickled_model
from src import Vectoriser, Detector


######################
# Training functions #
######################

def split_dataset(dataset, languages, use_seed):
    """ Split a given dataset (.csv) into training and test sets

    --- 50000 sentences per language are assumed in the dataset
    --- test set size can be defined in the 'config.py' file

    """

    # Read the given dataset into a pandas DataFrame
    train = pd.read_csv(dataset)
    # Create a new Dataframe to hold the test set
    test = pd.DataFrame()
    # For each language, append a given sample size of rows from the training set to the test set (seed if required)
    for lang, seed in zip(languages, list(range(1, len(languages)+1))):
        test = test.append(train.loc[train["language"] == lang].sample(n=config["test_size"], random_state=seed if use_seed == True else None))
    # Drop the rows in the test set from the training set 
    train = train.drop(train.index[test.index.values])
    # Seed numpy's random function, if required
    if use_seed == True:
        np.random.seed(0)
    # Shuffle the training data and return the desired sample siaze (as defined in the .config.py file)
    train = train.reindex(np.random.permutation(train.index.values))
    train = train.sample(n=config["train_size"], random_state=100 if use_seed == True else None)
    # Return the training and test sets
    return train, test


def train(languages, name, data=None, seed=False):
    """ Train a language detection model """

    # If an existing data file was provided, split it into training and test data
    if data != None:
        train, test = split_dataset(os.path.join(".", data), languages, seed)
    # Otherwise, split the default datset into training and test data
    else:
        train, test = split_dataset(config["dataset"], languages, seed)
    # Define destination folders in the 'models' directory, and create them if necessary
    destination = create_directory("models/{}".format(name))
    # Store the training and test data in the model's destination folder
    train.to_csv(os.path.join("./models", name, "train.csv"), index=False)
    test.to_csv(os.path.join("./models", name, "test.csv"), index=False)
    # Instantiate a Naive Bayes classifier
    clf = MultinomialNB()
    # Create a progress bar to track the progress of the training
    pbar = pyprind.ProgBar(len(train))
    X, y = [], []
    # For each row in the training data...
    for index, train_row in train.iterrows():
        # Append
        X.append(train_row["sentence"] if type(train_row["sentence"]) == str else "")
        y.append(train_row["language"])
        if len(X)%1000 == 0:
            X_vect = Vectoriser(gram_size=config["grams"]).transform(X)
            clf.partial_fit(X_vect, y, classes=languages)
            X, y = [], []
        pbar.update()
    # Store the trained model in its destination folder
    save_pickled_model(clf, destination)


#####################
# Utility functions #
#####################

def load_model_from_path(model_path):
    """ Load a model from a given path """

    if not os.path.exists(model_path):
        raise FileNotFoundError("The given model path could not be found.")
    model = load_pickled_model(model_path)
    return model


def detect_language(input_text, model_path):
    detector = Detector(gram_size=config["grams"])
    language = detector.detect_language(input_text, model_path)
    print(language)


####################
# Model evaluation #
####################

def evaluate(model_path):
    """ Evaluate a given language detection model, using the test data generated when it was trained """

    model = load_model_from_path(model_path)
    # Load in the model's test set and replace any 'NaN' fields with empty strings
    test = pd.read_csv(os.path.join(".", model_path, "test.csv"))
    test["sentence"] = test["sentence"].fillna("")
    # Extract the input sentences (vectorised) and language labels from the test set
    X = Vectoriser(gram_size=config["grams"]).transform(test["sentence"].values)
    y = test["language"].values
    # Calculate and print the model's score on the test set
    test_acc = model.score(X, y)
    print("\nAccuracy for language detection model '{}':\n{}".format(os.path.basename(model_path), "-"*int(41+0)))
    print(test_acc)
    # Calculate and print the model's score on each language individually
    for lang in test["language"].unique():
        lang_test = test.loc[test["language"] == lang]
        X = vect.transform(lang_test["sentence"].values)
        y = lang_test["language"].values
        lang_acc = model.score(X, y)
        print("Accuracy for '{}': {}".format(lang, lang_acc))


##################
# Main functions #
##################

def parse_training_arguments(arguments):
    """ Parse command line arguments (for training) """
    
    parser = argparse.ArgumentParser(description="lang_detector.py - Train a language detection model")
    optional = parser._action_groups.pop()
    required = parser.add_argument_group("required arguments")
    required.add_argument("-t", "--train", help="Train language detection model", action="store_true")
    optional.add_argument("-l", "--languages", help="Languages to include", nargs="+")
    optional.add_argument("-n", "--name", help="Name for the model currently being trained")
    optional.add_argument("-d", "--data", help="Name of pre-processed training and test sets to use for training")
    optional.add_argument("-s", "--seed", help="Perform deterministic training (when using existing data)", action="store_true", default=False)
    parser._action_groups.append(optional)
    return(parser.parse_args())


def parse_detection_arguments(arguments): 
    parser = argparse.ArgumentParser(description="lang-detector.py - Detect language for given input text")
    optional = parser._action_groups.pop()
    required = parser.add_argument_group("required arguments")
    required.add_argument("-d", "--detect",
                          help="Detect the language of a given input text", action="store_true")
    required.add_argument("-i", "--input",
                          help="Input text (string)", required=True)
    optional.add_argument("-m", "--model",
                          help="Language model to use for detection")
    parser._action_groups.append(optional)
    return(parser.parse_args())


def parse_evaluation_arguments(arguments):
    """ Parse command line arguments (for evaluation) """
    
    parser = argparse.ArgumentParser(description="lang_detector.py - Evaluate a pre-trained language detection model")
    optional = parser._action_groups.pop()
    required = parser.add_argument_group("required arguments")
    required.add_argument("-e", "--evaluate", help="Evaluate a pre-trained language detection model", action="store_true")
    required.add_argument("-m", "--model", help="Language model to use for detection")
    return(parser.parse_args())


if __name__ == "__main__":
    """ Train a model, evaluate a model, or use a model to detect target, depending on parsed arguments """

    args = sys.argv[1:] 
    # If input is being passed via standard input (as opposed to arguments), pass the standard input to the detect function
    if len(args) == 0 and not sys.stdin.isatty():
        print(detect_language(sys.stdin.read(), config["default_model"]))
    else:
        if len(args) != 0: 
            # If only one argument has been passed and it is a string, pass it to the detect function
            if len(args) == 1 and os.path.isfile(args[0]) != True and args[0].startswith("-") != True:
                print(detect_language(sys.stdin.read(), config["default_model"]))
            # If the first argument was '-t/--train'
            elif args[0] in ["-t", "--train"]:
                arguments = parse_training_arguments(args)
                # If no name was given, use the current date and time
                if arguments.name == None:
                    arguments.name = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
                # Use the given languages, or all default languages from the configuration file
                languages = sorted(arguments.languages) if arguments.languages != None else sorted(config["languages"].keys())
                # Train a language detection model
                train(languages, arguments.name, arguments.data, arguments.seed)
            elif args[0] in ["-d", "--detect"]:
                arguments = parse_detection_arguments(args)
                detect_language(arguments.input,
                                os.path.join(".", os.path.normpath(arguments.model))
                                if arguments.model is not None else config["default_model"])
                print(detect_language(arguments.input, os.path.join(".", os.path.normpath(arguments.model)) if arguments.model != None else config["default_model"]))
            # If the first argument was '-e/--evaluate'
            elif args[0] in ["-e", "--evaluate"]:
                arguments = parse_evaluation_arguments(args)
                # Evaluate the given language detection model
                evaluate(os.path.normpath(arguments.model))
            else:
                print("ERROR: First argument must specify training (-t/--train), detection (-d/--detect), or evaluation (-e/--evaluate)")
        else:
            print("ERROR: No arguments given. For help, please see the 'lang_detector' README")