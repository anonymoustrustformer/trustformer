# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 09:56:59 2024

@author: aliab
"""
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention, LinformerMultiHeadAttention
from PositionWiseFeedForward import PositionWiseFeedForward
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # self.self_attn = LinformerMultiHeadAttention(d_model, num_heads,256,50)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x