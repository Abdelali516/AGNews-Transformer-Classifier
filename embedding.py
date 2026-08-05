import torch 
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    def __init__(self,vocab_size, d_model):
        super().__init__()
        self.emdedding=nn.Embedding(num_embeddings=vocab_size,embedding_dim=d_model)
        self.d_model=d_model

    def forward(self, token_ids):
        return self.emdedding(token_ids)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()

        pe=torch.zeros(max_seq_len,d_model)
        pos=torch.arange(0,max_seq_len).unsqueeze(1)

        div_term=torch.exp(
            torch.arange(0,d_model,2) *(-math.log(10000.0)/d_model)
        )

        pe[:, 0::2]=torch.sin(pos*div_term)
        pe[:, 1::2]=torch.cos(pos*div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len=x.size(1)
        return x + self.pe[:seq_len, :]

