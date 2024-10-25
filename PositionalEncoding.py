# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 09:53:12 2024

@author: aliab
"""
import torch
import torch.nn as nn
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float16).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2,dtype=torch.float16) * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        # Set the seed
        self.set_seed(42)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


