# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) and this project uses [Semantic Versioning](http://semver.org/).

# [0.2.3] - 2020-2-18
### Added
 - `run_gpt2.bash` script to download models, encode texts, and train a GPT-2 model
 - exact library versions since Tensorflow 2 breaks oh so many things in GPT-2, yet it is not pinned
### Changed
 - all texts in `data/texts` are now GPT-2-ready by ending in `<|endoftext|>`
### Fixed
 - the mess that was the GPT-2 imports - geez!

# [0.2.2] - 2020-2-18
### Added
 - loss history is stored in the attribute `loss_history_`
 - `ReduceLROnPlateau` learning rate scheduler to `learner.train` to allow for automated fine-tuning of models during a regular training cycle
### Changed
 - the Ranger optimizer in `shakespeare_ai.externals.ranger.py`, used as the optimizer in `learner.py`
 - `iterations` is now referred to as `batches` in the verbose output of `learner.train`
 - GPT-2 source code is now stored in `shakespeare_ai.externals`
### Fixed
 - reported loss is not the running average over the epoch rather than from the latest batch

# [0.2.1] - 2020-2-18
### Fixed
 - removed unnecessary spaces before punctuation and ensured correct sentence capitalization when returning output in `predict`
 - verbose output during training now displays the thing it is actually saying it is displaying

# [0.2.0] - 2020-2-17
### Added
 - cloned GPT-2 model code
 - bash script to run the GPT-2 on the Shakespeare data

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
