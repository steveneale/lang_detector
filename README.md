# lang_detector

![version](https://img.shields.io/badge/version-0.1-important.svg) [![Travis CI](https://travis-ci.org/steveneale/lang_detector.svg?branch=master)](https://travis-ci.org/steveneale/lang_detector) [![Coverage Status](https://coveralls.io/repos/github/steveneale/lang_detector/badge.svg?branch=develop)](https://coveralls.io/github/steveneale/lang_detector?branch=develop)

*lang_detector* is a program for detecting the language of input texts, and for training and evaluating the models used for detection.


## Dependencies

*lang_detector* has been developed and tested on [Ubuntu](https://www.ubuntu.com/), and so these instructions should be followed with this in mind.

*lang_detector* is written in [Python](https://www.python.org/), and so a recent version of *python3* should be downloaded before using it. Downloads for *python* can be found at https://www.python.org/downloads/ (version 3.5.1 recommended).

The following *python* libraries will be needed to run *lang_detector* components:
* [pandas](https://pandas.pydata.org/)
* [NumPy](http://www.numpy.org/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [PyPrind](https://pypi.org/project/PyPrind/)


## Usage

With *python* installed, *lang_detector* can then be run from the command line.

In all examples, `*PATH*` refers to the path leading from the current directory to the directory in which the *lang_detector* folder is located.


## Detecting language

### Passing command line options to lang_detector

In *Linux*, using the `-d` / `--detect` option will enable the language of a string of text to be detected using a given model and choosing to use a bigram or trigram-based language profile:

For example, to detect the language of 'the quick brown fox jumped over the lazy dog', using a pre-trained model called 'test_model':

```bash
python3 *PATH*/lang_detector/lang_detector.py -i "The quick brown fox jumped over the lazy dog" -m models/test_model
```

#### Required arguments

##### -d/--detect

##### -i/--input [*string*]

A string of text for which to detect the language (enclosed in quotation marks). 

#### Optional arguments

*N.B.*: If the `-m` / `--model` option is not provided, the default language model (`models/default`) will be used.

##### -m/--model [*string*]

A pre-trained language detection model to be used.


### Passing a string of text to lang-detector

Alternatively, a single text string (enclosed in quotation marks) can be passed as an argument to *lang_detector*. This will return the most likely language using the default language model (`models/default`). For example, in *Linux*:

```bash
python3 *PATH*/lang_detector/lang_detector.py "The quick brown fox jumped over the lazy dog"
```

### Taking input text from standard input

*lang_detector* also accepts text passed via standard input. This will return the most likely language using the default language model (`models/default`). For example, in *Linux*:

```bash
echo "The quick brown fox jumped over the lazy dog" | python3 *PATH*/lang_detector/lang_detector.py
```

```bash
cat example.txt | python3 *PATH*/lang_detector/lang_detector.py
```

```bash
python3 *PATH*/lang_detector/lang_detector.py < example.txt
```


## Default model performance

The following is an evaluation of *lang_detector*'s default model, which can be found in `models/default`. The model was trained using the `data/default_data.csv` file, which contains 50000 sentences each taken from the English, French, Spanish, Portuguese, Italian, and German sides of the Europarl parallel corpus (created using `data/process_data.py`).

The model has been trained on 45000 sentences from each language, and evaluated on 5000 segments from each language. The training and test sets used in the model can be found in `models/default/train.csv` and `models/default/test.csv`.

| Language | Sentences | % Correct |
| ------ | ------ | ------ |
| Spanish | 5000 | 98.42% |
| Portuguese | 5000 | 98.66% |
| French | 5000 | 98.86% |
| English | 5000 | 99.26% |
| German | 5000 | 99.5% |
| Italian | 5000 | 99.5% |
| ------ | ------ | ------ |
| TOTAL | 30000 | 99.03% |


## Tests

*lang_detector* is complemented by unit tests, located in the `tests` directory. To run them, type the following from the *lang_detector*'s root directory:

```bash
python3 -m unittest discover -v -s tests
```