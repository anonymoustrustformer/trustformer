# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 09:58:28 2024

@author: aliab
"""
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention, LinformerMultiHeadAttention
from PositionWiseFeedForward import PositionWiseFeedForward
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # self.self_attn = LinformerMultiHeadAttention(d_model, num_heads,256, 50)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        # self.cross_attn = LinformerMultiHeadAttention(d_model,  num_heads, 256, 50)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x