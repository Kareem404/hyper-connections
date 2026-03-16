import torch 
import torch.nn as nn
import numpy as np
from .hyper_connections import mHC_attn, mHC_mlp

class MultiHeadAttention(nn.Module):
    """
    A GPT-style Multi-Head Attention implementation
    args:-
    - num_heads(int): Number of attention heads
    - d_model(int): Embedding dimension
    - dropout(float): the dropout rate
    """
    def __init__(
        self, 
        num_heads=4, 
        d_model=128, 
        dropout=0.1
    ):
        super().__init__()
        self.n = num_heads
        self.d_model = d_model
        self.dk = self.d_model//self.n

        scale = 0.02
        self.Wq = nn.Parameter(torch.randn(self.d_model, self.d_model) * scale, requires_grad=True) # [d_model, d_model]
        self.Wk = nn.Parameter(torch.randn(self.d_model, self.d_model) * scale, requires_grad=True) # [d_model, d_model]
        self.Wv = nn.Parameter(torch.randn(self.d_model, self.d_model) * scale, requires_grad=True) # [d_model, d_model]
        self.Wo = nn.Parameter(torch.randn(self.d_model, self.d_model) * scale, requires_grad=True) # [d_model, d_model]

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
    def forward(self, H):
        # H.shape = [b, T, d_model] where T is seq_len
        b, T, _ = H.shape

        Q = H @ self.Wq # [b, T, d_model]
        K = H @ self.Wk # [b, T, d_model]
        V = H @ self.Wv # [b, T, d_model]

        Q = Q.view(b, T, self.n, self.dk).transpose(1, 2) # [b, n, T, dk]
        K = K.view(b, T, self.n, self.dk).transpose(1, 2) # [b, n, T, dk]
        V = V.view(b, T, self.n, self.dk).transpose(1, 2) # [b, n, T, dk]

        mask = torch.tril(torch.ones(T, T, device=H.device)).unsqueeze(0).unsqueeze(0)

        scores = ((Q @ K.transpose(dim0=2, dim1=3))/np.sqrt(self.dk)).masked_fill(mask == 0, float('-inf'))

        attn = self.attn_dropout(torch.softmax(scores, dim=3)) @ V # [b, n, T, dk]

        # concat heads
        attn = attn.transpose(1, 2).contiguous().view(b, T, self.n * self.dk) # [b, T, d_model]

        return self.resid_dropout(attn @ self.Wo)
    
class MLP(nn.Module):
    """
    A GPT-style Multi-Head Mulit-Layer Preceptron implementation

    args:-
    - d_model(int): Embedding dimension
    - dropout(float): the dropout rate
    """
    def __init__(
        self, 
        d_model: int = 128, 
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model

        self.layer1 = nn.Linear(
            in_features=d_model, 
            out_features=4*d_model
        )

        self.gelu = nn.GELU()

        self.layer2 = nn.Linear(
            in_features=4*d_model, 
            out_features=d_model
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        args:-
        - x: x.shape = [b, T, d_model]
        """
        x = self.gelu(self.layer1(x)) # [b, T, 4*d_model]

        x = self.layer2(x) # [b, T, d_model]

        return self.dropout(x)
    

class TransformerBlock(nn.Module):
    """
        A GPT-style transformer layer:

        args:-
        - seq_len(int): The maximum sequence length or context
        - d_model(int): Embedding dimension 
        - num_heads(int): Number of attention heads per transformer layer
        - dropout(float): Dropout (between 0-1)
        - hc(bool): decide whether to use hyper connections or not
        - expansion_rate(int): how many copies of the input should be made. Redundant if hc = False 
    """
    def __init__(
        self, 
        d_model, 
        num_heads: int=4, 
        dropout: float=0.1, 
        seq_len: int=64, 
        hc = False, 
        expansion_rate: int=4, 
    ):
        super().__init__()
        self.hc = hc
        self.d_model = d_model
        self.num_heads = num_heads

        if hc:
            self.n = expansion_rate
            self.T = seq_len
        else:
            self.n = None
            self.T = None

        if not hc:
            self.layer_norm1 = nn.LayerNorm(
                self.d_model
            )
            self.layer_norm2 = nn.LayerNorm(
                self.d_model
            )

        self.attn = MultiHeadAttention(
            num_heads=num_heads, 
            d_model=self.d_model,
            dropout=dropout
        )
        
        if hc:
            self.mhc_attn = mHC_attn(
                self.attn, 
                expansion_rate=self.n, 
                d=self.d_model, 
                T=self.T
            )

        self.mlp = MLP(
            d_model=self.d_model, 
            dropout=dropout
        )

        if hc:
            self.mhc_mlp = mHC_mlp(
                self.mlp, 
                expansion_rate=self.n, 
                d=self.d_model
            )
    
    def forward(self, H):
        if not self.hc:
            H = H + self.attn(self.layer_norm1(H)) # [b, T, d_model]
            H = H + self.mlp(self.layer_norm2(H))
            return H # [b, T, d_model]
        else:
            b, T, n, d_model = H.shape

            H = self.mhc_attn(H) # [b*T, n, d_model]
            H = self.mhc_mlp(H) # [b*T, n, d_model]
            H = H.view(b, T, n, d_model) 

            return H # [b, T, n, d_model]

class GPT(nn.Module):
    """
        A GPT-2 style architecture:
        
        args:-
        - vocab_size(int): Vocab Size
        - seq_len(int): The maximum sequence length or context
        - d_model(int): Embedding dimension 
        - n_layers(int): Number of transformer layers
        - num_heads(int): Number of attention heads per transformer layer
        - dropout(float): Dropout (between 0-1)
        - hc(bool): decide whether to use hyper connections or not
        - expansion_rate(int): how many copies of the input should be made. Redundant if hc = False
    """
    def __init__(
        self, 
        vocab_size: int, 
        seq_len: int=64, 
        d_model: int=128, 
        n_layers: int=4,
        num_heads: int=4,
        dropout: float=0.1, 
        hc: bool = False, 
        expansion_rate: int = 4
    ):
        super().__init__()

        self.T = seq_len
        self.V = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_heads = num_heads

        self.hc = hc
        self.n = expansion_rate if self.hc else None

        self.token_embd = nn.Embedding(
            num_embeddings=self.V, 
            embedding_dim=self.d_model
        )

        self.pos_embd = nn.Embedding(
            num_embeddings=self.T,
            embedding_dim=self.d_model 
        )

        self.transformers = nn.ModuleList()
        for _ in range(n_layers):
            self.transformers.append(
                TransformerBlock(
                    d_model=self.d_model,
                    seq_len=self.T,
                    num_heads=self.num_heads,
                    expansion_rate=self.n,
                    dropout=dropout,
                    hc=self.hc
                )
            )
        
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        args:-
        - x: x.shape = [b, T]
        """

        b, T = x.shape

        token_embd = self.token_embd(x) # [b, T, d_model]
        
        positions = torch.arange(0, T, device=x.device).unsqueeze(0)

        pos_embd = self.pos_embd(positions) # [b, T, d_model]

        H = token_embd + pos_embd # [b, T, d_model]

        
        H = H.unsqueeze(2).expand(b, T, self.n, self.d_model).contiguous() if self.hc else H

        # apply transformers
        for layer in self.transformers:
            H = layer(H) # [b, T, n, d_model] if hc else [b, T, d_model]

        H = H.sum(dim=2) if self.hc else H # [b, T, d_model]
        H = self.layer_norm(H)

        E = self.token_embd.weight # [V, d_model]

        logits = H @ E.T # [b, T, V]

        return logits