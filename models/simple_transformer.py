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
    def __init__(self, embed_dim, context_len, dropout=0.1):
        super(PositionalEncodingLayer, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embed_dim = embed_dim
        self.context_len = context_len
        pos_encoding = self.construct_pos_encoding(self.context_len)
        self.register_buffer('pos_encoding', pos_encoding)

        self.dropout = nn.Dropout(p=dropout)

    def construct_pos_encoding(self, context_len):
        pos_encoding = torch.zeros(context_len, self.embed_dim).to(self.device)
        position = torch.arange(0, context_len, dtype=torch.float).unsqueeze(1).to(self.device)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * (-math.log(10000.0) / self.embed_dim)).to(self.device)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0)
        return pos_encoding

    def forward(self, x, adaptive_encoding=False):
        """
        Args:
            x (torch.tensor): input that needs to have pos embedding; 
                shape (batch_size, max_sequence_length, hidden_dim)
        Returns:
            embedding (torch.tensor): positional encoded input based on 'Attention Is All You Need' paper
        """
        if adaptive_encoding:
            _, L, _ = x.shape
            pos_encoding = self.construct_pos_encoding(L)
            embedding = x + pos_encoding[:, :x.shape[1]]
        else:
            embedding = x + self.pos_encoding[:, :x.shape[1]]
        return self.dropout(embedding)

class TransformerBlock(nn.Module):
    '''
    Implements a single transformer block, as described in the original
    Transformer paper.
    '''

    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_heads = num_heads

        # Initialize the position encoding
        self.hidden_dim = hidden_dim
        self.pos_encoding_layer = PositionalEncodingLayer(hidden_dim, 5000, dropout)        

        # Initialize the Multihead Attention and associated LayerNorm
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention_drop = nn.Dropout(p=dropout)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)

        # Initialize the feedforward layer and associated layernorm
        self.forward_norm = nn.LayerNorm(hidden_dim)
        self.forward_drop = nn.Dropout(p=dropout)
        self.forward_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, mask):
        '''
        Performs a forward pass of the TransformerBlock
        
        Args:
            x: The projected embeddings of the input sequence with shape (batch_size, seq_len, hidden_dim)
        '''
        batch_size, seq_len, hidden_dim = x.shape

        x = x.to(self.device)
        x = self.pos_encoding_layer(x)

        # Multihead Attention expects shape (seq_len, batch_size, hidden)
        x = x.permute(1, 0, 2)

        # We permute the attention output back to (batch_size, seq_len, hidden)
        mha_x = self.attention(x, x, x, attn_mask=mask)[0]
        mha_x = self.attention_norm(x + self.attention_drop(mha_x))

        # Perform the forward propagation
        f_x = F.relu(self.forward_proj(mha_x))
        f_x = self.forward_norm(mha_x + self.forward_drop(f_x))

        # We need to swap the shape back to (batch_size, seq_len, hidden_dim)
        f_x = f_x.permute(1, 0, 2)

        return f_x
        
class SimpleTransformer(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_blocks, num_heads, vocab_size):
        super(SimpleTransformer, self).__init__()
        
        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize the token and position embeddings
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(self.vocab_size, embed_dim, sparse=False)
        self.pos_embedding = nn.Embedding(3, embed_dim, sparse=False)

        # Initialze the projection to the hidden dim. Note that we need to double the embed_dim, because we
        # concatenate the token and positional embeddings
        self.proj = nn.Conv1d(2*embed_dim, hidden_dim, 1)

        # Initialize the transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(hidden_dim, num_heads) for _ in range(num_blocks)])

        # Initialize the final forward layer
        self.forward_proj = nn.Linear(hidden_dim, self.vocab_size)

        # Initialize the attention mask
        self.src_mask = None

    def get_mask(self, sz):
        '''
        Produces an attention mask for the given size
        '''
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask    

    def forward(self, token_ids):
        '''
        Performs the forward pass.

        Args:
           token_ids: Token ids with size (batch_size, sequence_length)
        '''
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

        # Perform the projection onto hidden_dim
        h = self.proj(full_embeds.permute(1, 2, 0)) # (batch_size, embed_dim, seq_len)

        # Produce the mask
        h = h.permute(0, 2, 1) # (batch_size, seq_len, embed_dim)
        if self.src_mask is None or self.src_mask.size(0) != seq_len:
            self.src_mask = self.get_mask(seq_len).to(self.device)

        # Apply the transformer blocks
        for block in self.blocks:
            h = block(h, self.src_mask)

        # Compute the final projection
        output = self.forward_proj(h)
        return output.permute(1, 0, 2)

        
        
