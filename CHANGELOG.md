# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) and this project uses [Semantic Versioning](http://semver.org/).

# [0.1.1] - 2020-2-17
### Added
 - ability to make a multi-layer LSTM with `num_layers` argument
### Changed
 - tokenization is done using `nltk.tokenize.word_tokenize`, reducing vocabulary to almost half the size

# [0.1.0] - 2020-2-17
### Added
 - `prepare.py` code to input a training dataset and prepare it for modeling
 - `model.py` code to define a PyTorch RNN
 - `learner.py` wrapper over `model.py` that nicely handles training, predicting, saving, and loading a model

# [0.0.0] - 2020-2-16
### Added
 - initial repo setup
