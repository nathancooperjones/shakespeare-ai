import torch
import torch.nn as nn


class RNNModule(nn.Module):
    """
    LSTM for text generation.

    Parameters
    ----------
    n_vocab: int
        The number of words in the vocabulary
    seq_size: int
        Length of sequence of words sent to the LSTM
    embedding_size: int
        Embedding size for each word in the vocab
    lstm_size: int
        The number of features in the hidden state of the LSTM

    """
    def __init__(self, n_vocab, seq_size, embedding_size, lstm_size):
        super(RNNModule, self).__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.lstm = nn.LSTM(embedding_size,
                            lstm_size,
                            batch_first=True)
        self.dense = nn.Linear(lstm_size, n_vocab)

    def forward(self, x, prev_state):
        """
        Generate output from the LSTM.

        Parameters
        ----------
        x: tensor
            Input to the LSTM
        prev_state: tuple of tensors
            Previous state of the model with format
            1st element: Initial hidden state for each element in the batch
            2nd element: Initial cell state for each element in the batch

        Returns
        ----------
        x: tensor
            Output from the LSTM
        state: tuple of tensors
            Current state of the model with format:
            1st element: Hidden state for each element in the batch
            2nd element: Cell state for each element in the batch

        """
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.dense(output)

        return logits, state

    def zero_state(self, batch_size):
        """
        Return zeroed-out hidden and cell states for the LSTM.

        Parameters
        ----------
        batch_size: int
            Batch size for the states

        Returns
        ----------
        zero-state: tuple of tensors
            Zeroed-out state of the model with format:
            1st element: Hidden state for each element in the batch, all zeroes
            2nd element: Cell state for each element in the batch, all zeroes

        """
        return (torch.zeros(1, batch_size, self.lstm_size),
                torch.zeros(1, batch_size, self.lstm_size))
