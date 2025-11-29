import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, batch_first=False):
        super(PositionalEncoding, self).__init__()
        self.batch_first = batch_first
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.batch_first:
            # x is [batch, seq, d_model]
            # pe is [max_len, 1, d_model]
            # We need [1, seq, d_model] to broadcast correctly
            return x + self.pe[:x.size(1), :].transpose(0, 1)
        else:
            # x is [seq, batch, d_model]
            return x + self.pe[:x.size(0), :]

class CollatzTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, max_len=500):
        super(CollatzTransformer, self).__init__()
        
        # Input embedding for parity vector (0, 1, and -1 for padding)
        self.embedding = nn.Embedding(3, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, batch_first=True)
        
        # Enable batch_first=True for better performance
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Heads
        self.stopping_time_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Predict next parity bit (Autoregressive task)
        self.next_step_head = nn.Linear(d_model, 3) # 0, 1, 2(pad/end)

    def forward(self, src, src_key_padding_mask=None):
        # src: [batch_size, seq_len]
        
        # No transpose needed for batch_first=True
        # src = src.transpose(0, 1) 
        
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim) # [batch, seq, d_model]
        x = self.pos_encoder(x)
        
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask) # [batch, seq, d_model]
        
        # Pooling for global properties (stopping time)
        # Mean over sequence dimension (dim=1)
        pooled = output.mean(dim=1) # [batch, d_model]
        
        stopping_time_pred = self.stopping_time_head(pooled)
        next_step_logits = self.next_step_head(output) # [batch, seq, 3]
        
        return stopping_time_pred, next_step_logits
