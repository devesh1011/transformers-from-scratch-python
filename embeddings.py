import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.input_embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        embedding = self.input_embed(x)
        return embedding * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))


    def forward(self, x):
        return x + self.pe[:, :x.size(1)]