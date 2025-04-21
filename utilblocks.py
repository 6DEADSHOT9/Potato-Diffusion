import torch
from torch import nn
from torch.nn import functional as F
import math

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv2d_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)

        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x:torch.Tensor) -> torch.Tensor:

        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)

        x = self.conv2d_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)

        x = self.conv2d_2(x)

        residue = self.residual(residue)

        x = x + residue

        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self,n_heads: int, d_embed: int, in_proj_bias = True, out_proj_bias = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias = in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias = out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x:torch.Tensor, causal_mask = False) -> torch.Tensor:

        # (Batch size, Seq len, Dim(d_embed)) 
        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape

        interim_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        # (Batch size, Seq len, Dim(d_embed)) ->  (Batch size, Seq len, 3* Dim(d_embed)) 3 tensors of shape (Batch size, Seq len, Dim(d_embed))
        q, k, v = self.in_proj(x).chunk(3, dim = -1)

        # (Batch size, Seq len, Dim(d_embed)) -> (Batch size, Seq len, Heads, d_head) -> (Batch size, Heads, Seq len, d_head)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (Batch size, Heads, Seq len, d_head) @ (Batch size, Heads, d_head, Seq len) -> (Batch size, Heads, Seq len, Seq len)
        weights = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(weights).triu(1) # mask above the principle diagonal
            weights.masked_fill_(mask, -torch.inf)  # apply the mask if wanted

        weights /= math.sqrt(self.d_head)
        weights = F.softmax(weights, dim=-1)

        # (Batch size, Heads, Seq len, Seq len) @ (Batch size, Heads, Seq len, d_head) -> (Batch size, Heads, Seq len, d_head)
        out = weights @ v

        # (Batch size, Heads, Seq len, d_head) -> (Batch size, Seq len, Heads, d_head)
        out = out.transpose(1, 2)

        # (Batch size, Seq len, Heads, d_head) -> (Batch size, Seq len, d_embed)
        out = out.reshape(input_shape)

        out = self.out_proj(out)

        return out
    
class VaeAttentionBLock(nn.Module):
    def __init__(self, channels: int,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.groupnorm_1 = nn.GroupNorm(32, channels)
        self.attention = SelfAttentionBlock(1, channels)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:

        residue = x

        # (Batch size, Channels, Height, Width)
        n, c, h, w = x.shape

        # (Batch size, Channels, Height, Width) -> (Batch size, Channels, Height * Width) -> (Batch size, Height * Width, Channels)
        x = x.view(n, c, h*w).transpose(-1, -2)

        x = self.attention(x)

        # (Batch size, Height * Width, Channels) -> (Batch size, Channels, Height * Width) -> (Batch size, Channels, Height, Width)
        x = x.transpose(-1, -2).view(n, c, h, w)

        x = x + residue
        
        return x