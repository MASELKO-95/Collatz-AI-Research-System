import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class CollatzTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, max_len=500):
        super(CollatzTransformer, self).__init__()
        
        # Input embedding for parity vector (0, 1, and -1 for padding)
        # We shift -1 to 2 to make it a valid index if we use Embedding
        # Or we can just mask it. Let's assume 0=even, 1=odd, 2=pad
        self.embedding = nn.Embedding(3, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, dropout=0.1)
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
        # src: [seq_len, batch_size]
        # src values: 0, 1. Padding should be handled.
        
        # Map -1 to 2 for embedding if present, or assume input is already 0,1,2
        # We'll assume input is [batch, seq_len]
        src = src.transpose(0, 1) # [seq_len, batch]
        
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Pooling for global properties (stopping time)
        # We can take the mean or just the first token if we had a CLS token.
        # Let's take mean over non-padded elements.
        # For simplicity, just max pool or mean pool.
        pooled = output.mean(dim=0) # [batch, d_model]
        
        stopping_time_pred = self.stopping_time_head(pooled)
        next_step_logits = self.next_step_head(output) # [seq_len, batch, 3]
        
        return stopping_time_pred, next_step_logits.transpose(0, 1)
