#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
'process_data.py'

Process raw input files into .csv-formatted language data files, that can be used 
for training language models for 'lang_detector'.

2017 Steve Neale <steveneale3000@gmail.com>

"""

import sys
import os
import math
import random

import argparse
import pyprind

import pandas as pd


###################
# File processing #
###################

def stream_lines(path, skip_header=False):
    """ Stream lines from a given file path """

    # Open the given file
    with open(path, "r", encoding="utf-8") as file:
        # Skip the header if required
        if skip_header == True:
            next(file)
        # Yield lines from the given file one by one
        for line in file:
            yield line


def get_minibatch_lines(doc_stream, size):
    """ Retrieve a batch of lines from a given document stream """

    lines = []
    try:
        # For a given batch size...
        for _ in range(size):
            # Get the next line from the document stream and append it to the list of lines
            line = next(doc_stream)
            lines.append(line.strip())
    # If a 'StopIteraction' exception is raised...
    except StopIteration:
        # Return any lines already streamed, or None
        if len(lines) > 0:
            return lines
        else:
            return None
    # Return the streamed batch of lines
    return lines


def process_input_data(name, seed=False):
    """ Process input data from raw files into a CSV file

    --- Assumes data/ folder (where this file is located) contains one subfolder per language
    --- Folder names should ideally correspond to ISO 639-1 codes:
        --- see: https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
    --- Files in each language folder are assumed to be:
        --- Plain text, UTF-8
        --- Longer than 50,000 lines (as 50,000 lines are taken at random from each file) 

    """

    # Create a pandas DataFrame to store the processed language data
    df = pd.DataFrame(columns=["sentence", "source", "language"])
    # For each language in the current directory (should be data/)...
    for lang, seed in zip([x for x in os.listdir(".") if os.path.isdir(os.path.join(".", x))], list(range(1, len([x for x in os.listdir(".") if os.path.isdir(os.path.join(".", x))])+1))):
        print("\nProcessing resources in '{}':\n{}".format(lang, "-"*(29)))
        # For each file in the language directory...
        for file in os.listdir(os.path.join(".", lang)):
            # Open the file and count the number of lines
            with open(os.path.join(".", lang, file), "r", encoding="utf-8") as infile:
                line_count = len(infile.read().splitlines())
            print("{} lines from '{}'".format(line_count, file))
            # Seed the random function, if required
            if seed == True:
                random.seed(seed)
            # Pull 50000 sorted line IDs at random
            data_idx = sorted(random.sample(range(0, line_count), 50000))
            # Create a progress bar to monitor the progress of the file processing
            pbar = pyprind.ProgBar(math.ceil(line_count/1000))
            # Define the document stream from which to process lines
            doc_stream = stream_lines(path=os.path.join(".", lang, file))
            # For every 1000th of the number of lines in the file...
            for _ in range(0, math.ceil(line_count/1000)):
                # Retrieve a batch of 1000 lines from the file
                lines = get_minibatch_lines(doc_stream, size=1000)
                if not lines:
                    break
                # For each of the 1000 lines...
                for _line, line in enumerate(lines):
                    line_id = (_line+1) + (_*1000)
                    # If the current line is equal to the first line in the list of 50000 random line IDs
                    if line_id == data_idx[0]:
                        # Drop the first entry (current line ID) from the list of 50000 line IDs
                        data_idx = data_idx[1:] if len(data_idx) > 1 else data_idx
                        # Append the current line to the DataFrame
                        df = df.append({"sentence": line, "source": file, "language": lang}, ignore_index=True)
                pbar.update()
    # Write the DataFrame to CSV
    df.to_csv("./{}.csv".format(name), index=False)


##################
# Main functions #
##################

def parse_arguments(arguments):
    """ Parse command line arguments """
    
    parser = argparse.ArgumentParser(description="process_data.py - Process raw data from language specific folders into a .csv file")
    optional = parser._action_groups.pop()
    required = parser.add_argument_group("required arguments")
    required.add_argument("-n", "--name", help="Name for the language data file to be created", required=True)
    optional.add_argument("-s", "--seed", help="Collect samples from the raw files in a deterministic way (for reproducability)", action="store_true", default=False)
    parser._action_groups.append(optional)
    return(parser.parse_args())


if __name__ == "__main__":
    """ Process input data from raw files into a CSV file """
    
    # Parse command line arguments
    arguments = parse_arguments(sys.argv[1:])
    # Process input data
    process_input_data(arguments.name, arguments.seed)
