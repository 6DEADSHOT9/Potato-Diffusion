import torch
from torch import nn
from torch.nn import functional as F
from utilblocks import SelfAttentionBlock, CrossAttentionBlock


class TimeEmbedding(nn.Module):

    def __init__(self, n_embed, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.linear_1 = nn.Linear(n_embed, n_embed * 4)
        self.linear_2 = nn.Linear(n_embed * 4, n_embed * 4)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        
        # x: (1,320)

        x = self.linear_1(x)  # (1,320) -> (1,1280)
        x = F.silu(x)

        x = self.linear_2(x)  # (1,1280) -> (1,1280)

        return x  # (1,1280)


class SwitchSequential(nn.Sequential):

    def forward (self, x:torch.Tensor, context:torch.Tensor, time:torch.Tensor) -> torch.Tensor:

        for layer in self:

            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        
        return x

class Upsample(nn.Module):

    def __init__(self, channels:int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)

        return x


class UNET_ResidualBlock(nn.Module):

    def __init__(self, in_channels:int, out_channels:int, n_time:int = 1280, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x:torch.Tensor, time:torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, in_channels, Height/16, Width/16)
        # time: (1, 1280)

        residue = x

        x = self.groupnorm_feature(x)
        x = F.silu(x)
        x = self.conv_feature(x)

        time = F.silu(time)
        time = self.linear_time(time)

        merged = x + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)

        merged = self.conv_merged(merged)

        output = merged + self.residual(residue)
        
        return output

class UNET_AttentionBlock(nn.Module):

    def __init__(self, n_head:int, n_embed, d_context:int = 768, *args, **kwargs):
        super().__init__(*args, **kwargs)

        channels = n_head * n_embed

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttentionBlock(n_head, channels, in_proj_bias = False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttentionBlock(n_head, channels, d_context, in_proj_bias = False)
        self.layernorm_3 = nn.LayerNorm(channels)

        self.linear_geglu_1 = nn.Linear(channels, 4*channels*2)
        self.linear_geglu_2 = nn.Linear(4*channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x:torch.Tensor, context:torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, Channels, Height, Width)
        residue_long = x

        x = self.groupnorm(x)
        x = self.conv_input(x)

        n, c, h, w = x.shape

        x = x.view(n, c, h*w)   # (Batch_size, Channels, Height*Width)
        x = x.transpose(-1, -2) # (Batch_size, Height*Width, Channels)

        residue_short = x

        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x = x + residue_short

        residue_short = x
        
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x = x + residue_short

        residue_short = x

        x = self.layernorm_3(x)

        x = self.linear_geglu_1(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x = x + residue_short

        x = x.transpose(-1, -2) # (Batch_size, Channels, Height*Width)
        
        x = x.view(n, c, h, w)

        x = self.conv_output(x)
        x = x + residue_long

        return x