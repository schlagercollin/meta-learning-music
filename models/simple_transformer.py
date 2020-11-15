# File: simple_transformer
# ------------------------
# Implements a basic transformer based language model.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncodingLayer(nn.Module):
    """
    Adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Uses the same positional encoding technique as described in the `Attention Is All You Need` paper (sinusoidal)

    """
    def __init__(self, embed_dim, context_len):
        super(PositionalEncodingLayer, self).__init__()
        pos_encoding = torch.zeros(context_len, embed_dim)
        position = torch.arange(0, context_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        """

        Args:
            x (torch.tensor): input that needs to have pos embedding; 
                shape (batch_size, max_sequence_length, hidden_dim)
        Returns:
            embedding (torch.tensor): positional encoded input based on 'Attention Is All You Need' paper
        """

        embedding = x + self.pos_encoding[:, :x.shape[1]]
        #embedding = F.dropout(embedding, p=c.POS_ENCODE_DROP_PROB, training=self.training)
        return embedding

class TransformerBlock(nn.Module):
    '''
    Implements a single transformer block, as described in the original
    Transformer paper.
    '''

    def __init__(self, hidden_dim, num_heads, context_len):
        super(TransformerBlock, self).__init__()
        self.num_heads = num_heads

        # Initialize the position encoding
        self.hidden_dim = hidden_dim
        self.pos_encoding_layer = PositionalEncodingLayer(hidden_dim, context_len)        

        # Initialize the Multihead Attention and associated LayerNorm
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)

        # Initialize attention mask
        self.register_buffer("attention_mask", torch.tril(torch.ones(context_len, context_len)))

        # Initialize the feedforward layer and associated layernorm
        self.forward_norm = nn.LayerNorm(hidden_dim)
        self.forward_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        '''
        Performs a forward pass of the TransformerBlock
        
        Args:
            x: The projected embeddings of the input sequence with shape (batch_size, seq_len, hidden_dim)
        '''
        batch_size, seq_len, hidden_dim = x.shape

        x = self.pos_encoding_layer(x)

        # Multihead Attention expects shape (seq_Len, batch_size, hidden)
        x = x.permute(1, 0, 2)

        # We permute the attention output back to (batch_size, seq_len, hidden)
        mha_x = self.attention(x, x, x, attn_mask=self.attention_mask)[0]
        mha_x = self.attention_norm(x + mha_x)


        # Perform the forward propagation
        f_x = F.relu(self.forward_proj(mha_x))
        f_x = self.forward_norm(mha_x + f_x)

        # We need to swap the shape back to (batch_size, seq_len, hidden_dim)
        f_x = f_x.permute(1, 0, 2)

        return f_x
        
class SimpleTransformer(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_blocks, num_heads, context_len, vocab_size):
        super(SimpleTransformer, self).__init__()
        

        # NOTE: the context_len is in total number of tokens. We need to divide this by 3 
        # (to account for the fact that we process pitch, duration, and advance simulatenously) 
        # and then subtract 1 (to account for the fact that we omit the final token  in the input)
        self.context_len = (context_len // 3) - 1

        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize the token and position embeddings
        self.vocab_size = vocab_size
        self.pitch_embedding = nn.Embedding(self.vocab_size, embed_dim, sparse=True)
        self.duration_embedding = nn.Embedding(self.vocab_size, embed_dim, sparse=True)
        self.advance_embedding = nn.Embedding(self.vocab_size, embed_dim, sparse=True)

        # Initialze the projection to the hidden dim. Note that we need to triple the embed_dim, because we
        # concatenate the pitch, duration, and advance embeddings
        self.proj = nn.Conv1d(3*embed_dim, hidden_dim, 1)

        # Initialize the transformer blocks
        self.blocks = [TransformerBlock(hidden_dim, num_heads, self.context_len) for _ in range(num_blocks)]

        # Initialize the final forward layer
        self.forward_proj = nn.Linear(hidden_dim, self.vocab_size*3)

    def forward(self, token_ids, pos_idx_start=0):
        '''
        Performs the forward pass.

        Args:
           token_ids: Token ids with size (batch_size, sequence_length)
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

        # Get the overall embedding
        full_embeds = torch.cat((pitch_embeds, duration_embeds, advance_embeds), dim=2) # (seq_len, batch_size, 3*embed_dim)

        # Perform the projection onto hidden_dim
        h = self.proj(full_embeds.permute(1, 2, 0)) # (batch_size, embed_dim, seq_len)

        # Apply the transformer blocks
        h = h.permute(0, 2, 1) # (batch_size, seq_len, embed_dim)
        for block in self.blocks:
            h = block(h)

        # Output of proj is (batch, seq_len, 3*vocab_size)
        projected = self.forward_proj(h)

        # First we permute into (seq_len, batch, 3*vocab_size)
        projected = projected.permute(1, 0, 2)

        # And then we reshape to (seq_len, batch, 3, vocab_size)
        projected = projected.reshape(seq_len, batch_size, 3, self.vocab_size)

        # Compute the final projection
        return projected

        
        
