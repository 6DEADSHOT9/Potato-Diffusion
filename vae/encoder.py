import torch
from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # (Batch_size, Channels, Height, Width) -> (Batch_size, 128, Height, Width)
        nn.Conv2d(3, 128, kernel_size=3, padding=1)
