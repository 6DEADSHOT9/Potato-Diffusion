import torch
from torch import nn
from torch.nn import functional as F
from utilblocks import ResidualBlock, VaeAttentionBLock


class Decoder(nn.Sequential):

    def __init__(self, *args, **kwargs):
        super().__init__(

            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            ResidualBlock(512, 512),

            VaeAttentionBLock(512),

            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),

            ResidualBlock(512, 512),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),
            nn.SiLU(),
            
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, 4, Height/8, Width/8)

        x /= 0.18215 #idk why

        for module in self:
            x = module(x)
        
        # (Batch_size, 3, Height, Width)
        return x