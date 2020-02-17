from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from shakespeare_ai.model import RNNModule
from shakespeare_ai.prepare import get_data_from_file


class ShakespeareLearner():
    """
    Wrapper over the LSTM model to simply train, predict, save, and load.

    Parameters
    ----------
    train_file: str
        Path to text file to train LSTM with
    batch_size: int
        Number of training examples utilized in one iteration (default 16)
    seq_size: int
        Length of sequence of words sent to the model (default 32)
    embedding_size: int
        Embedding size for each word in the vocab (default 64)
    lstm_size: int
        The number of features in the hidden state of the LSTM (default 64)
    lr: float
        Learning rate for Adam optimizer (default 0.01)
    gradients_norm: float
        Max norm of the gradients (default 5)
    initial_words: list of strings
        Initial string seed for prediction text. If `None`, will choose a random word
        in the vocabulary to begin the predicted text (default None)
    predict_top_k: int
        How many of the top predicted words to randomly pick during each word generation (default 5)
    checkpoint_path: int
        Path to save model checkpoints (default 'checkpoint')
    checkpoint_frequency: int
        After how many interations to checkpoint the model. If < 0, will not
        checkpoint the model (default 1000)
    verbose: bool
        Print progress / predictions after every epoch (default True)

    """
    def __init__(self,
                 train_file,
                 batch_size=16,
                 seq_size=32,
                 embedding_size=64,
                 lstm_size=64,
                 lr=0.01,
                 gradients_norm=5,
                 initial_words=None,
                 predict_top_k=5,
                 checkpoint_path='checkpoint',
                 checkpoint_frequency=1000,
                 verbose=True):
        self.train_file = train_file
        self.batch_size = batch_size
        self.seq_size = seq_size
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size
        self.lr = lr
        self.gradients_norm = gradients_norm
        self.initial_words = initial_words
        self.predict_top_k = predict_top_k
        self.checkpoint_path = checkpoint_path
        self.checkpoint_frequency = checkpoint_frequency
        self.verbose = verbose

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        (self.int_to_vocab,
         self.vocab_to_int,
         self.n_vocab,
         self.in_text,
         self.out_text) = get_data_from_file(self.train_file,
                                             self.batch_size,
                                             self.seq_size)

        self.model = RNNModule(self.n_vocab,
                               self.seq_size,
                               self.embedding_size,
                               self.lstm_size)
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, num_epochs=50):
        """
        Train the LSTM model.

        Parameters
        ----------
        num_epochs: int
            Number of epochs to train the model for (default 50)

        """
        iteration = 0
        for epoch in range(num_epochs):
            batches = self._get_batches(self.in_text, self.out_text, self.batch_size, self.seq_size)
            hidden_state, cell_state = self.model.zero_state(self.batch_size)

            # Transfer data to GPU
            hidden_state = hidden_state.to(self.device)
            cell_state = cell_state.to(self.device)
            for input, expected_output in batches:
                iteration += 1

                # Tell it we are in training mode
                self.model.train()

                # Reset all gradients
                self.optimizer.zero_grad()

                # Transfer data to GPU, if available
                input = torch.tensor(input).to(self.device)
                expected_output = torch.tensor(expected_output).to(self.device)

                # Send the data to the model
                logits, (hidden_state, cell_state) = self.model(input, (hidden_state, cell_state))
                # Need to transpose the output to match the shape of `expected_output`
                loss = self.criterion(logits.transpose(1, 2), expected_output)

                hidden_state = hidden_state.detach()
                cell_state = cell_state.detach()

                loss_value = loss.item()

                # Perform back-propagation
                loss.backward(retain_graph=True)

                # Update the network's parameters
                self.optimizer.step()
                loss.backward()

                # Prevent the model's parameters from exploding by normalizing
                _ = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.gradients_norm)

                self.optimizer.step()

                if (self.checkpoint_frequency > 0) and (iteration % self.checkpoint_frequency == 0):
                    Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)
                    self.save(f'{self.checkpoint_path}/model-{iteration}.pth')

            if self.verbose:
                print('Epoch: {}/{}'.format(epoch, 200),
                      'Iteration: {}'.format(iteration),
                      'Loss: {}'.format(loss_value))
                self.predict(self.initial_words)

    def _get_batches(self, in_text, out_text, batch_size, seq_size):
        """
        Convert the text into formats suitable for batches based on `batch_size`.

        Parameters
        ----------
        in_text: np.array
            Array of integers of size `seq_size` for model input
        out_text: np.array
            Array of integers of size `seq_size` for evaluating model output
        batch_size: int
            Number of training examples utilized in one iteration
        seq_size: int
            Length of sequence of words sent to the model

        Returns
        ----------
        Yields tuple containing batches of `in_text` and `out_text` of length `seq_size`.

        """
        num_batches = np.prod(in_text.shape) // (seq_size * batch_size)

        for i in range(0, num_batches * seq_size, seq_size):
            yield in_text[:, i:i+seq_size], out_text[:, i:i+seq_size]

    def predict(self, start=None, length=100):
        """
        Generate predictions from the model.

        Parameters
        ----------
        start: string
            Initial string seed for prediction text. If `None`, will choose a random word
            in the vocabulary to begin the predicted text (default None)
        length: int
            Length of the outputted text (default 100)

        Returns
        ----------
        output: string
            Generated text outputted from the LSTM model

        """
        self.model.eval()

        hidden_state, cell_state = self.model.zero_state(1)
        hidden_state = hidden_state.to(self.device)
        cell_state = cell_state.to(self.device)

        if start is None:
            try:
                sentence = [self.int_to_vocab[np.random.randint(0, len(self.int_to_vocab))]]
            except KeyError as oops:
                print('Unknown vocabulary:', oops)
                print('Using random start seed instead.')
                sentence = start.split()
        else:
            sentence = start.split()

        for word in sentence:
            ix = torch.tensor([[self.vocab_to_int[word]]]).to(self.device)
            output, (hidden_state, cell_state) = self.model(ix, (hidden_state, cell_state))

        _, top_ix = torch.topk(output[0], k=self.predict_top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])

        sentence.append(self.int_to_vocab[choice])

        for _ in range(length):
            ix = torch.tensor([[choice]]).to(self.device)
            output, (hidden_state, cell_state) = self.model(ix, (hidden_state, cell_state))

            _, top_ix = torch.topk(output[0], k=self.predict_top_k)
            choices = top_ix.tolist()
            choice = np.random.choice(choices[0])
            sentence.append(self.int_to_vocab[choice])

        print(' '.join(sentence))

    def save(self, filename):
        """
        Save the model's state_dict.

        Parameters
        ----------
        filename: str
            Path to save the model to (ending in '.pth')

        """
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        """
        After instantiating an instance of the `ShakespeareLearner` class, loads
        the previously saved state_dict at `filename`.

        Parameters
        ----------
        filename: str
            Path to load the model from

        """
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()
