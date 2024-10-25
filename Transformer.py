# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 10:10:44 2024

@author: aliab
"""
import torch
import torch.nn as nn
from PositionalEncoding import PositionalEncoding
from EncoderLayer import EncoderLayer
from DecoderLayer import DecoderLayer
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, transformer_id):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        #print("hello")
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.transformer_id = transformer_id
        self.set_seed(42)
    # def generate_mask(self, src, tgt):
    #     src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    #     if tgt.shape[0] <= 0:
    #         tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
    #         seq_length = tgt.size(0)
    #     else:
    #         tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
    #         seq_length = tgt.size(1)
    #     nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
    #     tgt_mask = tgt_mask & nopeak_mask
    #     return src_mask, tgt_mask
    def generate_mask(self, src, tgt):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        src_mask = src != 0
        tgt_mask = tgt != 0
        
        seq_length = tgt.size(1)
        nopeak_mask = torch.triu(torch.ones(seq_length, seq_length).to(device = device), diagonal=1).bool().to(device = device)
    
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)
        tgt_mask = tgt_mask.unsqueeze(1).unsqueeze(3) & nopeak_mask
    
        return src_mask, tgt_mask

    

    # def forward(self, src, tgt):
    #     src_mask, tgt_mask = self.generate_mask(src, tgt)
    #     src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
    #     print("src: ", src)
    #     print("tgt: ", tgt)
    #     if len(tgt.data)==0: 
    #         print("Ohooi")
    #         tgt=torch.tensor([[0 for i in range(src.shape[1])]])
    #         print("tgt: ", tgt)
    #     tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

    #     enc_output = src_embedded
    #     for enc_layer in self.encoder_layers:
    #         enc_output = enc_layer(enc_output, src_mask)

    #     dec_output = tgt_embedded
    #     for dec_layer in self.decoder_layers:
    #         dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

    #     output = self.fc(dec_output)
    #     return output
    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set the seed

    def forward(self, src, tgt=None):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output

    # def generate(self, input_sequence, max_length=50, temperature=1.0):
    #     # Set the model to evaluation mode
    #     self.eval()
    #     end_of_sequence_token = 1  # Update with your end-of-sequence token index
    
    #     # Initialize the generated output
    #     generated_output = []
    #     generated_output = torch.tensor(generated_output)

    #     with torch.no_grad():
    #         # Perform greedy decoding
    #         for _ in range(max_length):
    #             # Forward pass to generate the next token
    #             output = self.forward(input_sequence, generated_output)
    #             next_token = output.argmax(dim=-1)[:,-1]
                
    #             # Append the next token to the generated output
    #             generated_output.append(next_token.item())
                
    #             # Stop generation if the end-of-sequence token is generated
    #             if next_token == end_of_sequence_token:
    #                 break
                
    #             # Append the generated token to the input sequence for the next step
    #             input_sequence = torch.cat([input_sequence, next_token.unsqueeze(1)], dim=-1)
        
    #     return generated_output


    def generate(self, input_sequence, tokenizer, max_length=50, temperature=1.0):
        # Set the model to evaluation mode
        self.eval()
        end_of_sequence_token = tokenizer.eos_token_id

        # Initialize the generated output
        generated_output = []

        with torch.no_grad():
            # Perform greedy decoding
            for _ in range(max_length):
                # Forward pass to generate the next token
                output = self.forward(input_sequence)
                next_token = output.argmax(dim=-1)[:,-1]
                
                # Append the next token to the generated output
                generated_output.append(next_token.item())
                
                # Stop generation if the end-of-sequence token is generated
                if next_token == end_of_sequence_token:
                    break
                
                # Append the generated token to the input sequence for the next step
                input_sequence = torch.cat([input_sequence, next_token.unsqueeze(1)], dim=-1)
        
        return generated_output
    
    def getTransformerId(self):
        return self.transformer_id

