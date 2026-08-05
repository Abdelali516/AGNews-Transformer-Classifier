import torch
import torch.nn as nn

from embedding import TokenEmbedding, PositionalEncoding
from attention_block import TransformerBlock


class NewsTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_seq_len: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
 
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

      
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, token_ids, attention_mask):
        mask = None
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)

   
        x = self.token_embedding(token_ids)          # (batch, seq_len, d_model)
        x = self.positional_encoding(x)               # (batch, seq_len, d_model)
        x = self.dropout(x)

        all_attn_weights = []
        for block in self.blocks:
            x, attn_weights = block(x, mask)
            all_attn_weights.append(attn_weights)

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            summed = (x * mask_expanded).sum(dim=1)                # (batch, d_model)
            counts = mask_expanded.sum(dim=1).clamp(min=1e-9)       # avoid divide-by-zero
            pooled = summed / counts                                 # (batch, d_model)
        else:
            pooled = x.mean(dim=1)  

        logits = self.classifier(pooled)  # (batch, num_classes)

        return logits, all_attn_weights


def make_padding_mask(token_ids, pad_id):
    return (token_ids != pad_id).long()

