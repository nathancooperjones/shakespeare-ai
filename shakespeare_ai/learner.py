from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from shakespeare_ai.model import RNNModule
from shakespeare_ai.prepare import get_batches, get_data_from_file


class ShakespeareLearner():
    """TODO."""
    def __init__(self,
                 train_file,
                 batch_size=16,
                 seq_size=32,
                 embedding_size=64,
                 lstm_size=64,
                 lr=0.01,
                 gradients_norm=5,
                 initial_words=['How', 'do', 'I'],
                 predict_top_k=5,
                 checkpoint_path='checkpoint'):
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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        (self.int_to_vocab,
         self.vocab_to_int,
         self.n_vocab,
         self.in_text,
         self.out_text) = get_data_from_file(self.train_file,
                                             self.batch_size,
                                             self.seq_size)

        self.net = RNNModule(self.n_vocab,
                             self.seq_size,
                             self.embedding_size,
                             self.lstm_size)
        self.net = self.net.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def train(self):
        """TODO."""
        iteration = 0
        for e in range(50):
            batches = get_batches(self.in_text, self.out_text, self.batch_size, self.seq_size)
            state_h, state_c = self.net.zero_state(self.batch_size)

            # Transfer data to GPU
            state_h = state_h.to(self.device)
            state_c = state_c.to(self.device)
            for x, y in batches:
                iteration += 1

                # Tell it we are in training mode
                self.net.train()

                # Reset all gradients
                self.optimizer.zero_grad()

                # Transfer data to GPU
                x = torch.tensor(x).to(self.device)
                y = torch.tensor(y).to(self.device)

                logits, (state_h, state_c) = self.net(x, (state_h, state_c))
                loss = self.criterion(logits.transpose(1, 2), y)

                state_h = state_h.detach()
                state_c = state_c.detach()

                loss_value = loss.item()

                # Perform back-propagation
                loss.backward(retain_graph=True)

                # Update the network's parameters
                self.optimizer.step()

                loss.backward()

                _ = torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.gradients_norm)

                self.optimizer.step()

                if iteration % 100 == 0:
                    print('Epoch: {}/{}'.format(e, 200),
                          'Iteration: {}'.format(iteration),
                          'Loss: {}'.format(loss_value))

                if iteration % 1000 == 0:
                    self.predict(self.initial_words)
                    Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)
                    torch.save(self.net.state_dict(),
                               f'{self.checkpoint_path}/model-{iteration}.pth')

    def predict(self, words):
        """TODO."""
        self.net.eval()

        state_h, state_c = self.net.zero_state(1)
        state_h = state_h.to(self.device)
        state_c = state_c.to(self.device)
        for w in words:
            ix = torch.tensor([[self.vocab_to_int[w]]]).to(self.device)
            output, (state_h, state_c) = self.net(ix, (state_h, state_c))

        _, top_ix = torch.topk(output[0], k=self.predict_top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])

        words.append(self.int_to_vocab[choice])

        for _ in range(100):
            ix = torch.tensor([[choice]]).to(self.device)
            output, (state_h, state_c) = self.net(ix, (state_h, state_c))

            _, top_ix = torch.topk(output[0], k=self.predict_top_k)
            choices = top_ix.tolist()
            choice = np.random.choice(choices[0])
            words.append(self.int_to_vocab[choice])

        print(' '.join(words))
