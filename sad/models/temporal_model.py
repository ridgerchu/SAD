import torch
import torch.nn as nn

from sad.layers.temporal import Bottleneck3D, TemporalBlock
from sad.layers.convolutions import ConvBlock, Bottleneck, SpikingDeepLabHead


class TemporalModelIdentity(nn.Module):
    def __init__(self, in_channels, receptive_field):
        super().__init__()
        self.receptive_field = receptive_field
        self.out_channels = in_channels

    def forward(self, x):
        return x