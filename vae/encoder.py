import torch
from torch import nn
from torch.nn import functional as F
from utilblocks import ResidualBlock, VaeAttentionBLock

class Encoder(nn.Sequential):

    def __init__(self, *args, **kwargs):
        super().__init__(
        
        # (Batch_size, Channels, Height, Width) -> (Batch_size, 128, Height, Width)
        nn.Conv2d(3, 128, kernel_size = 3, padding = 1),

        ResidualBlock(128, 128),

        nn.Conv2d(128, 128, kernel_size = 3, stride = 2, padding = 0),
        
        ResidualBlock(128, 256),
        ResidualBlock(256, 256),

        nn.Conv2d(256, 256, kernel_size = 3, stride = 2, padding = 0),

        ResidualBlock(256, 512),
        ResidualBlock(512, 512),

        nn.Conv2d(512, 512, kernel_size = 3, stride = 2, padding = 0),

        ResidualBlock(512, 512),
        ResidualBlock(512, 512),
        ResidualBlock(512, 512),

        VaeAttentionBLock(512),

        nn.GroupNorm(32, 512),

        nn.SiLU(),

        nn.Conv2d(512, 8, kernel_size = 3, padding = 1),
        nn.Conv2d(8, 8, kernel_size = 3, padding = 1),
        
        )
    
    def forward(self, x:torch.Tensor, noise:torch.Tensor) -> torch.Tensor:

        # x: (Batch_size, Channels, Height, Width)
        # noise: (Batch_size, out channels, Height/8, Width/8)

        for module in self:
            if getattr(module, 'stride', None) == 2:
                x = F.padd(x, (0, 1, 0, 1))
            x = module(x)
        
        # x: (Batch_size, 8, Height/8, Width/8) -> 2*(Batch_size, 4, Height/8, Width/8)
        mean, log_var = torch.chunk(x, 2, dim = 1)

        log_var = torch.clamp(log_var, -30, 20)

        var = torch.exp(log_var)

        std = torch.sqrt(var)

        x = mean + std * noise

        x *= 0.18215 #idk why

        return x