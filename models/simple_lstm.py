import os
import math
import torch
import pickle
import datetime
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F

class SimpleLSTM(nn.Module):
    def __init__(self,
                 embed_dim,
                 hidden_dim,
                 vocab_size=128,
                 num_layers=2,
                 dropout=0.5,
                 lr=1e-3):

        super(SimpleLSTM, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.vocab_size = vocab_size

        # Encodes the (pitch, dur, adv) tuples
        self.token_embedding = nn.Embedding(self.vocab_size, embed_dim)

        # Encodes the position within each tuple, i.e. [0, 1, 2, 0, 1, 2, ...]
        self.pos_embedding = nn.Embedding(3, embed_dim)

        # NOTE: input dimension is 2 * embed_dim because we have embeddings for both
        # the token IDs and the positional IDs
        self.lstm = nn.LSTM(2 * embed_dim, hidden_dim, num_layers=num_layers, dropout=dropout)

        self.proj = nn.Linear(hidden_dim, self.vocab_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)


    def forward(self, token_ids):
        '''
        Args:
            token_ids: size is (batch_size, sequence_length)
        '''
        batch_size, seq_len = token_ids.shape

        token_embeds = self.token_embedding(token_ids)

        # Permute into (seq_len, batch, embed_size)
        token_embeds = token_embeds.permute(1, 0, 2)

        # The position ids are just 0, 1, and 2 repeated for as long
        # as the sequence length
        pos_ids = torch.tensor([0, 1, 2]).repeat(batch_size, math.ceil(seq_len/3))[:, :seq_len]
        pos_ids = pos_ids.to(self.device)
        pos_embeds = self.pos_embedding(pos_ids)
        pos_embeds = pos_embeds.permute(1, 0, 2)

        full_embeds = torch.cat((token_embeds, pos_embeds), dim=2)

        lstm_out, _ = self.lstm(full_embeds)

        projected = self.proj(lstm_out)

        return projected