import torch
import torch.nn as nn
from attention import MultiHeadAttention
from feed_forward import FeedForwardNet


class DecoderBlock(nn.Module):
    def __init__(self, d_model, h, dropout=0.1) -> None:
        super(DecoderBlock, self).__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, h)
        self.cross_attn = MultiHeadAttention(d_model, h)
        self.feed_forward = FeedForwardNet(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tar_mask):
        attn_output = self.masked_self_attn(x, x, x, tar_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x