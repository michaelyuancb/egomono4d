"""Adapted from https://github.com/milesial/Pytorch-UNet/tree/master/unet"""
"""Copied and updated from https://github.com/jphdotam/Unet3D/blob/main/unet3d.py"""
import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F


class UNet3D(nn.Module):
    def __init__(self, input_channels, feature_channels: list, kernel_size: int, 
                 groups: int=1,
                 trilinear=True, use_ds_conv=False):
        """A simple 3D Unet, adapted from a 2D Unet from https://github.com/milesial/Pytorch-UNet/tree/master/unet
        Arguments:
          n_channels = number of input channels; 3 for RGB, 1 for grayscale input
          n_classes = number of output channels/classes
          width_multiplier = how much 'wider' your UNet should be compared with a standard UNet
                  default is 1;, meaning 32 -> 64 -> 128 -> 256 -> 512 -> 256 -> 128 -> 64 -> 32
                  higher values increase the number of kernels pay layer, by that factor
          trilinear = use trilinear interpolation to upsample; if false, 3D convtranspose layers will be used instead
          use_ds_conv = if True, we use depthwise-separable convolutional layers. in my experience, this is of little help. This
                  appears to be because with 3D data, the vast vast majority of GPU RAM is the input data/labels, not the params, so little
                  VRAM is saved by using ds_conv, and yet performance suffers."""
        super(UNet3D, self).__init__()
        # _channels = (32, 64, 128, 256, 512)
        self.channels = feature_channels
        self.n_channels = len(self.channels)
        assert self.n_channels > 1
        self.kernel_size = kernel_size
        self.convtype = DepthwiseSeparableConv3d if use_ds_conv else nn.Conv3d

        in_channels = input_channels
        self.inc = DoubleConv(in_channels, self.channels[0], self.kernel_size, groups=groups, conv_type=self.convtype)

        self.down_module = nn.ModuleList([])
        for i in range(self.n_channels-1):
            down = Down(self.channels[i], self.channels[i+1], self.kernel_size, groups=groups, conv_type=self.convtype)
            self.down_module.append(down)

        self.fc = nn.Conv3d(self.channels[-1], self.channels[-1], kernel_size=(1,1,1))

        self.up_module = nn.ModuleList([])
        for i in reversed(range(1, self.n_channels)):
            up = Up(self.channels[i], self.channels[i-1], kernel_size, groups=groups, trilinear=trilinear)
            self.up_module.append(up)
        
        self.out = OutConv(self.channels[0], in_channels)


    def forward(self, x):
        feat = self.inc(x)                           # torch.Size([1, 256, 5, 37, 49])
        res_feat_list = [feat]
        for i in range(self.n_channels-1):
            feat = self.down_module[i](feat)         # torch.Size([1, 384, 5, 18, 24])
            res_feat_list.append(feat)               # torch.Size([1, 512, 5, 9, 12])
                                                     # torch.Size([1, 768, 5, 4, 6])
        feat = self.fc(feat)                         # torch.Size([1, 768, 5, 4, 6])
        # pdb.set_trace()
        for i in range(self.n_channels-1):
            res_x = res_feat_list[-i-2]
            feat = self.up_module[i](feat, res_x) + res_x
           
        u_feat = self.out(feat) + x
        return u_feat


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size, groups=1, conv_type=nn.Conv3d, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        # self.double_conv = nn.Sequential(
        #     conv_type(in_channels, mid_channels, kernel_size=kernel_size, groups=groups, padding=kernel_size//2),
        #     nn.BatchNorm3d(mid_channels),
        #     nn.ReLU(inplace=True),
        #     conv_type(mid_channels, out_channels, kernel_size=kernel_size, groups=groups, padding=kernel_size//2),
        #     nn.BatchNorm3d(out_channels),
        #     nn.ReLU(inplace=True)
        # )
        self.double_conv = nn.Sequential(
            conv_type(in_channels, out_channels, kernel_size=kernel_size, padding=1, groups=groups),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size, groups=1, conv_type=nn.Conv3d):
        super().__init__()
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool3d((1, 2, 2)),
            DoubleConv(in_channels, out_channels, kernel_size, groups=groups, conv_type=conv_type)
        )

    def forward(self, x):
        return self.avgpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size, groups=1, trilinear=True):
        super().__init__()

        # if trilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size, groups=groups)
        else:
            raise ValueError("Only support for trilinear is True.")

    def forward(self, x, x_target):
        # pdb.set_trace()
        _, _, f, h, w = x_target.shape
        x = F.interpolate(x, (f, h, w), mode='trilinear', align_corners=True)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, kernels_per_layer=1):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.depthwise = nn.Conv3d(nin, nin * kernels_per_layer, kernel_size=kernel_size, padding=padding, groupss=nin)
        self.pointwise = nn.Conv3d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out