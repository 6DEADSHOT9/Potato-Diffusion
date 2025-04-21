import torch
import torch.nn as nn
import torch.nn.functional as F
from utilblocks import SelfAttentionBlock

class CLIPEmbedding(nn.Module):

    def __init__(self, n_vocab:int, n_embed:int, n_tokens:int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.token_embeddings = nn.Embedding(n_vocab, n_embed)
        self.position_embeddings = nn.Parameter(torch.zeros(n_tokens, n_embed))

    def forward(self, tokens:torch.Tensor) -> torch.Tensor:
        # (Batch_size, Seq_len) -> (Batch_size, Seq_len, dim)

        x = self.token_embeddings(tokens)

        x += self.position_embeddings

        return x
    
class CLIPLayer(nn.Module):

    def __init__(self, n_head:int, n_embed:int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.attention = SelfAttentionBlock(n_head, n_embed)

        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)

        self.linear1 = nn.Linear(n_embed, n_embed * 4)
        self.linear2 = nn.Linear(n_embed * 4, n_embed)

    def forward(self, x:torch.Tensor) -> torch.Tensor:

        residue = x

        x = self.layer_norm1(x)
        x = self.attention(x, causal_mask=True)
        x = x + residue

        residue = x
        x = self.layer_norm2(x)
        x = self.linear1(x)
        x = torch.sigmoid(1.702 * x) # Quick GELU activation the reason for 1.702 is unknown

        x = self.linear2(x)
        x = x + residue

        return x

class CLIP(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.Module([
            CLIPLayer(12, 768) for _ in range(12)
            ])
        self.layer_norm = nn.LayerNorm(768)

    def forward(self, tokens:torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # (Batch_size, Seq_len) -> (Batch_size, Seq_len, dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        # (Batch_size, Seq_len, dim)
        output = self.layer_norm(state)

        return output