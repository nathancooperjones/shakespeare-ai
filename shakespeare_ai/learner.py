from pathlib import Path
import re

import nltk
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tqdm

from shakespeare_ai.externals.ranger import RangerVA
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
    hidden_size: int
        The number of features in the hidden state of the LSTM (default 64)
    num_layers: int
        Number of recurrent layers for the LSTM (default 1)
    lr: float
        Learning rate for Adam optimizer (default 0.01)
    gradients_norm: float
        Max norm of the gradients (default 5)
    initial_words: list of strings
        Initial string seed for prediction text. If `None`, will choose a random word
        in the vocabulary to begin the predicted text (default None)
    temperature: int
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
                 hidden_size=64,
                 num_layers=1,
                 lr=1e-2,
                 gradients_norm=5,
                 initial_words=None,
                 temperature=5,
                 checkpoint_path='checkpoint',
                 checkpoint_frequency=1000,
                 verbose=True):
        self.train_file = train_file
        self.batch_size = batch_size
        self.seq_size = seq_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.gradients_norm = gradients_norm
        self.initial_words = initial_words
        self.temperature = temperature
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
                               self.hidden_size,
                               self.num_layers)
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = RangerVA(self.model.parameters(), lr=self.lr)

        self.epochs = 0
        self.batches = 0
        self.loss_history_ = list()

        self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def train(self,
              num_epochs=50,
              scheduler_patience=0,
              scheduler_factor=0.1,
              scheduler_threshold=1e-4):
        """
        Train the LSTM model.

        Parameters
        ----------
        num_epochs: int
            Number of epochs to train the model for (default 50)
        scheduler_patience: int
            number of epochs of a plateaued learning rate before reducing the learning
            rate by a factor of `scheduler_factor` for `torch.optim.lr_scheduler.ReduceLROnPlateau`
            (default 0)
        scheduler_factor: int
            when validation loss plateuas after `scheduler_patience`
            epochs, factor of which the learning rate should be reduced
            by for `torch.optim.lr_scheduler.ReduceLROnPlateau` (default 0.1)
        scheduler_threshold: float
            threshold for measuring the new optimum, to only focus on significant changes
            for `torch.optim.lr_scheduler,ReduceLROnPlateau` (default 1e-4)

        """
        scheduler = ReduceLROnPlateau(self.optimizer,
                                      mode='min',
                                      factor=scheduler_factor,
                                      patience=scheduler_patience,
                                      threshold=scheduler_threshold,
                                      verbose=True)

        try:
            # we don't need it writing a multiple lines per progress bump
            tqdm._instances.clear()
        except AttributeError:
            pass

        for epoch in tqdm.tqdm(range(num_epochs), position=0, unit=' epoch', desc=''):
            batches = self._get_batches(self.in_text, self.out_text, self.batch_size, self.seq_size)
            hidden_state, cell_state = self.model.zero_state(self.batch_size)

            # Transfer data to GPU
            hidden_state = hidden_state.to(self.device)
            cell_state = cell_state.to(self.device)

            avg_loss_list = list()
            for input, expected_output in batches:
                self.batches += 1

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

                avg_loss_list.append(loss.item())

                # Perform back-propagation
                loss.backward(retain_graph=True)

                # Update the network's parameters
                self.optimizer.step()
                loss.backward()

                # Prevent the model's parameters from exploding by normalizing
                _ = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.gradients_norm)

                self.optimizer.step()

                if (self.checkpoint_frequency > 0) and (self.batches % self.checkpoint_frequency == 0):
                    Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)
                    self.save(f'{self.checkpoint_path}/model-{self.batches}.pth')

            avg_loss = np.mean(np.array(avg_loss_list))
            self.loss_history_.append(avg_loss)
            scheduler.step(avg_loss, epoch=self.epochs)
            if self.verbose:
                print(
                    'Epoch: {0:^3}  Batches: {1:^11}  Loss: {2:^10.5f}'
                    .format(self.epochs, self.batches, avg_loss)
                )
                print(self.predict(self.initial_words))

            self.epochs += 1

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
            encoded_start = [np.random.randint(0, len(self.int_to_vocab))]
        else:
            try:
                encoded_start = [self.vocab_to_int[word] for word in start.split()]
            except KeyError as oops:
                print('Unknown vocabulary:', oops)
                print('Using random start seed instead.')
                encoded_start = [np.random.randint(0, len(self.int_to_vocab))]

        # reverse engineer the start to become the beginning of the predicted sentence
        sentence = [self.int_to_vocab[word_int] for word_int in encoded_start]

        for encoded_word in encoded_start:
            ix = torch.tensor([[encoded_word]]).to(self.device)
            output, (hidden_state, cell_state) = self.model(ix, (hidden_state, cell_state))

        _, top_ix = torch.topk(output[0], k=self.temperature)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])

        sentence.append(self.int_to_vocab[choice])

        for _ in range(length):
            ix = torch.tensor([[choice]]).to(self.device)
            output, (hidden_state, cell_state) = self.model(ix, (hidden_state, cell_state))

            _, top_ix = torch.topk(output[0], k=self.temperature)
            choices = top_ix.tolist()
            choice = np.random.choice(choices[0])
            sentence.append(self.int_to_vocab[choice])

        prediction = ' '.join(sentence)
        prediction = re.sub(r'\s([?.!,"\(\[\)\]](?:\s|$))', r'\1', prediction)
        prediction_sentences = self.sentence_tokenizer.tokenize(prediction)
        prediction_sentences = [sentence.capitalize() for sentence in prediction_sentences]
        prediction = ' '.join(prediction_sentences)

        return prediction

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
