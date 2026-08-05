import torch
import torch.nn as nn

class  MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=d_model//num_heads

        self.W_q=nn.Linear(d_model,d_model)
        self.W_k=nn.Linear(d_model,d_model)
        self.W_v=nn.Linear(d_model,d_model)
        self.W_o=nn.Linear(d_model,d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len,_=x.shape

        Q=self.W_q(x)
        K=self.W_k(x)
        V=self.W_v(x)

        Q=Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
        K=K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
        V=V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)

        scores=torch.matmul(Q,K.transpose(-2,-1)) / self.d_k**0.5

        if mask is not None:
            scores=scores.masked_fill(mask==0,float('-inf'))

        attention_weights=torch.softmax(scores,dim=-1)

        output=torch.matmul(attention_weights,V)

        output=output.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)

        output=self.W_o(output)

        return output, attention_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.Linear1=nn.Linear(d_model,d_ff)
        self.relu=nn.ReLU()
        self.Linear2=nn.Linear(d_ff,d_model)

    def forward(self, x):
        return self.Linear2(self.relu(self.Linear1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout: float=0.1):
        super().__init__()
        self.attention=MultiHeadSelfAttention(d_model,num_heads)
        self.norm1=nn.LayerNorm(d_model)
        self.ffn=FeedForward(d_model,d_ff)
        self.norm2=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self, x, mask=None):
        att_output, att_weights=self.attention(x, mask)
        x=self.norm1(x + self.dropout(att_output))

        ffn_output=self.ffn(x)
        x=self.norm2(x + self.dropout(ffn_output))

        return x, att_weights
