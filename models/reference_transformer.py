# File: reference_transformer
# ---------------------------
# Transformer language-model implementation borrowed
# from Pytorch. Solely uses Pytorch nn.Module calls.

import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import torch.nn.functional as F
import numpy as np

from models.simple_transformer import PositionalEncodingLayer

class ReferenceTransformer(nn.Module):
    '''
    Reference transformer word model implementation adapted
    from Pytorch.
    '''

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_blocks, num_heads, dropout=0.5):
        super(ReferenceTransformer, self).__init__()

        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize the token and position embeddings
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(self.vocab_size, embed_dim, sparse=False)
        self.pos_embedding = nn.Embedding(3, embed_dim, sparse=False)

        # Initialize the transformer
        self.src_mask = None
        self.pos_encoder = PositionalEncodingLayer(2 * embed_dim, 5000)
        encoder_layers = TransformerEncoderLayer(2 * embed_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_blocks)

        # Initialize the decoder
        self.decoder = nn.Linear(2 * embed_dim, self.vocab_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.token_embedding.weight, -initrange, initrange)
        nn.init.uniform_(self.pos_embedding.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def get_mask(self, sz):
        '''
        Produces an attention mask for the given size
        '''
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, token_ids):
        '''
        Perform the forward pass, as before. Expects token ids with size
        (batch_size, sequence_length)
        '''
        # Extract the size
        batch_size, seq_len = token_ids.shape
        token_ids = token_ids.to(self.device)

        # Perform the token embedding
        token_embeds = self.token_embedding(token_ids)
        token_embeds = token_embeds.permute(1, 0, 2) # (seq_len, batch_size, embed_dim)
        
        # Perform the position embedding
        pos_ids = torch.tensor([0, 1, 2]).repeat(batch_size, math.ceil(seq_len/3))[:, :seq_len]
        pos_ids = pos_ids.to(self.device)
        pos_embeds = self.pos_embedding(pos_ids)
        pos_embeds = pos_embeds.permute(1, 0, 2)

        # Get the overall embedding
        full_embeds = torch.cat((token_embeds, pos_embeds), dim=2) # (seq_len, batch_size, 2*embed_dim)
        full_embeds = self.pos_encoder(full_embeds.permute(1, 0, 2)).permute(1, 0, 2)

        # Construct the mask if none has been constructed or if there is a size mismatch
        if self.src_mask is None or self.src_mask.size(0) != len(full_embeds):
            mask = self.get_mask(len(full_embeds)).to(self.device)
            self.src_mask = mask
        output = self.transformer_encoder(full_embeds, self.src_mask)
        output = self.decoder(output)
        return output
        
