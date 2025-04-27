import torch
from torch import nn
from torch.nn import functional as F
from diffblocks import SwitchSequential, UNET_ResidualBlock, UNET_AttentionBlock, Upsample


class UNET(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Module([
            # (Batch_size, 4, Height/8, Width/8)

            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8,40)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8,40)),

            # (Batch_size, 320, Height/8, Width/8) -> (Batch_size, 320, Height/16, Width/16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2,  padding=1)),

            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8,80)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8,80)),

            # (Batch_size, 640, Height/16, Width/16) -> (Batch_size, 640, Height/32, Width/32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8,160)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8,160)),

            # (Batch_size, 1280, Height/32, Width/32) -> (Batch_size, 1280, Height/64, Width/64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, padding=1)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            # Batch_size, 1280, Height/64, Width/64) -> (Batch_size, 1280, Height/64, Width/64) 
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            
            UNET_AttentionBlock(8, 160),
            
            UNET_ResidualBlock(1280, 1280),
        )

        self.decoder = nn.ModuleList([

            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8,160)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8,160)),
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8,160), Upsample(1280)),
            
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8,80)),
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8,80)),
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8,80), Upsample(640)),

            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8,40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8,40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8,40)),
        
        ])

class UNET_OutputLayer(nn.Module):

    def __init__(self, in_channels:int, out_channels:int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
            
            x = self.groupnorm(x)
            x = F.silu(x)
            x = self.conv(x)
    
            return x