import torch
import torch.nn as nn
from embeddings import PositionalEncoding
from encoder import EncoderBlock
from decoder import DecoderBlock

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tar_vocab_size, d_model, h, num_layers, max_seq_len, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tar_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

        self.encoder_layers = nn.ModuleList([EncoderBlock(d_model, h, dropout) for _ in range(num_layers
        )])
        self.decoder_layers = nn.ModuleList([DecoderBlock(d_model, h, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tar_vocab_size)
        self.dropout = nn.Dropout(dropout)


    def generate_mask(self, src, tar):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tar_mask = (tar != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tar.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tar_mask = tar_mask & nopeak_mask
        return src_mask, tar_mask
    
    def forward(self, src, tar):
        src_mask, tar_mask = self.generate_mask(src, tar)
        src_embedded = self.dropout(self.positional_encoding((self.encoder_embedding(src))))
        tar_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tar)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tar_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tar_mask)

        output = self.fc(dec_output)
        return output