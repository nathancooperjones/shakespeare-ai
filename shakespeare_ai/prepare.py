from collections import Counter

import nltk
import numpy as np


def get_data_from_file(train_file, batch_size, seq_size):
    """
    Read data from `train_file` and generate translation dictionaries as well as
    data for the LSTM model consumption.

    Parameters
    ----------
    train_file: str
        Filename for the text file to train the model with
    batch_size: int
        Number of training examples utilized in one iteration
    seq_size: int
        Length of sequence of words sent to the model

    Returns
    ----------
    int_to_vocab_dict: dict
        Dictionary converting integers to their respective vocabulary words
    vocab_to_int_dict: dict
        Dictionary converting vocabulary words to their respective integers
    n_vocab: int
        Number of words in the vocabulary
    in_text: np.array
        Array of integers of size `seq_size` for model input
    out_text: np.array
        Array of integers of size `seq_size` for evaluating model output

    """
    nltk.download('punkt')

    with open(train_file, 'r') as f:
        text = f.read()
    text = nltk.tokenize.word_tokenize(text)

    # set up word to integer encoding system
    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)

    int_to_vocab_dict = {key: word for key, word in enumerate(sorted_vocab)}
    vocab_to_int_dict = {word: key for key, word in int_to_vocab_dict.items()}

    n_vocab = len(int_to_vocab_dict)
    print(f'Vocabulary size: {n_vocab:,} words.')

    # convert text to ints
    int_text = [vocab_to_int_dict[word] for word in text]
    num_batches = int(len(int_text) / (seq_size * batch_size))
    in_text = int_text[:(num_batches * batch_size * seq_size)]  # cut off text not in batch

    # shift data to the left by one so we have a target to predict
    out_text = np.zeros_like(in_text)
    out_text[:-1] = in_text[1:]  # shift `out_text` to the left by one of `in_text`
    out_text[-1] = in_text[0]  # replace last word of `out_text` with first of `in_text`

    # set up text for batches
    in_text = np.reshape(in_text, (batch_size, -1))
    out_text = np.reshape(out_text, (batch_size, -1))

    return int_to_vocab_dict, vocab_to_int_dict, n_vocab, in_text, out_text
