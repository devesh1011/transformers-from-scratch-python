import torch
import torch.nn as nn
from attention import MultiHeadAttention
from feed_forward import FeedForwardNet


class EncoderBlock(nn.Module):
    def __init__(self, d_model, h, dropout=0.1) -> None:
        super(EncoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, h)
        self.feed_forward = FeedForwardNet(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
