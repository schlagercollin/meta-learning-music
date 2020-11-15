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

        self.pitch_embedding = nn.Embedding(self.vocab_size, embed_dim, sparse=True)
        self.duration_embedding = nn.Embedding(self.vocab_size, embed_dim, sparse=True)
        self.advance_embedding = nn.Embedding(self.vocab_size, embed_dim, sparse=True)

        self.lstm = nn.LSTM(3 * embed_dim, hidden_dim, num_layers=num_layers, dropout=dropout)

        self.proj = nn.Linear(hidden_dim, self.vocab_size * 3)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)


    def forward(self, token_ids):
        '''
        Args:
            token_ids: size is (batch_size, 3, sequence_length)
        '''
        batch_size, _, seq_len = token_ids.shape

        pitch_ids, duration_ids, advance_ids = torch.split(token_ids, 1, dim=1)

        pitch_embeds = self.pitch_embedding(pitch_ids.squeeze(1))
        duration_embeds = self.duration_embedding(duration_ids.squeeze(1))
        advance_embeds = self.advance_embedding(advance_ids.squeeze(1))


        # Permute into (seq_len, batch, embed_size)
        pitch_embeds = pitch_embeds.permute(1, 0, 2)
        duration_embeds = duration_embeds.permute(1, 0, 2)
        advance_embeds = advance_embeds.permute(1, 0, 2)

        full_embeds = torch.cat((pitch_embeds, duration_embeds, advance_embeds), dim=2)

        lstm_out, _ = self.lstm(full_embeds)

        projected = self.proj(lstm_out)

        # We need to convert the output into shape (seq_len, batch_size, 3, vocab_size)
        projected = projected.reshape(seq_len, batch_size, 3, self.vocab_size)

        return projected