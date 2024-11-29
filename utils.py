import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()  
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)  
        normalized_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * normalized_x + self.bias



class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1) -> None:
        super().__init__()  
        self.dropout = dropout
        self.norm = LayerNormalization(d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, sub_layer):
        return self.norm(x + self.dropout_layer(sub_layer(x)))

