from collections import Counter

import numpy as np


def get_data_from_file(train_file, batch_size, seq_size):
    """TODO."""
    with open(train_file, 'r') as f:
        text = f.read()
    text = text.split()

    # set up word to integer encoding system
    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)

    int_to_vocab_dict = {key: word for key, word in enumerate(sorted_vocab)}
    vocab_to_int_dict = {word: key for key, word in int_to_vocab_dict.items()}

    n_vocab = len(int_to_vocab_dict)
    print(f'Vocabulary size: {n_vocab}.')

    # convert text to ints
    int_text = [vocab_to_int_dict[word] for word in text]
    num_batches = int(len(int_text) / (seq_size * batch_size))
    in_text = int_text[:(num_batches * batch_size * seq_size)]  # cut off text not in batch

    # shift data to the left by one so we have a target to predict
    out_text = np.zeros_like(in_text)
    out_text[:-1] = in_text[1:]  # shift `out_text` to the left by one of `in_text`
    out_text[-1] = in_text[0]  # replace last character of `out_text` with first of `in_text`

    # set up text for batches
    in_text = np.reshape(in_text, (batch_size, -1))
    out_text = np.reshape(out_text, (batch_size, -1))

    return int_to_vocab_dict, vocab_to_int_dict, n_vocab, in_text, out_text


def get_batches(in_text, out_text, batch_size, seq_size):
    """TODO."""
    num_batches = np.prod(in_text.shape) // (seq_size * batch_size)

    for i in range(0, num_batches * seq_size, seq_size):
        yield in_text[:, i:i+seq_size], out_text[:, i:i+seq_size]
