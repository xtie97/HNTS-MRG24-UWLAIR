from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.layers.factories import Conv


# CAM
class ChannelGate(nn.Module):
    def __init__(self, channels, r):
        super(ChannelGate, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(
                in_features=self.channels,
                out_features=self.channels // self.r,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=self.channels // self.r,
                out_features=self.channels,
                bias=True,
            ),
        )

    def forward(self, x):
        max = F.adaptive_max_pool3d(x, output_size=1)
        avg = F.adaptive_avg_pool3d(x, output_size=1)
        b, c, _, _, _ = x.size()
        linear_max = self.linear(max.view(b, c)).view(b, c, 1, 1, 1)
        linear_avg = self.linear(avg.view(b, c)).view(b, c, 1, 1, 1)
        output = linear_max + linear_avg
        output = torch.sigmoid(output) * x
        return output


# SAM
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self) -> None:
        super(SpatialGate, self).__init__()
        kernel_size = 7

        self.compress = nn.Sequential(
            # nn.InstanceNorm3d(input_channel, affine=True),
            # nn.ReLU(),
            ChannelPool(),  # max and mean, 2 channels, original code only uses ChannelPool()
        )

        self.spatial = Conv[Conv.CONV, 3](
            in_channels=6,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )

    def forward(self, x, mask):
        x_compress = self.compress(x)  # 2 channels, max and mean
        output = self.spatial(
            torch.cat((x_compress, mask), dim=1)
        )  # concatenate the mask (4 channels)
        output = torch.sigmoid(output) * x
        return output


# CBAM
class CBAM(nn.Module):
    def __init__(self, input_channel):
        super(CBAM, self).__init__()
        """
        Adapted from convolutional block attention module (CBAM) implementation
        https://github.com/Peachypie98/CBAM/blob/main/cbam.py
        """
        self.channels = input_channel
        # specify the parameters for CAM and SAM
        self.r = 16
        self.cam = ChannelGate(self.channels, self.r)
        self.sam = SpatialGate()

    def forward(self, x, mask):
        # apply CAM and SAM
        output = self.cam(x)
        output = self.sam(output, mask)

        return output + x
