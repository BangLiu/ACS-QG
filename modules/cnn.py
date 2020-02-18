"""
CNN modules.
"""
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """
    Depth-wise separable convolution uses less parameters
    to generate output by convolution.
    :Examples:
        >>> m = DepthwiseSeparableConv(300, 200, 5, dim=1)
        >>> input = torch.randn(32, 300, 20)
        >>> output = m(input)
    """

    def __init__(self, in_ch, out_ch, k, dim=1, bias=True):
        """
        :param in_ch: input hidden dimension size
        :param out_ch: output hidden dimension size
        :param k: kernel size
        :param dim: default 1. 1D conv or 2D conv
        :param bias: default False. Add bias or not
        """
        super().__init__()
        if dim == 1:
            self.depthwise_conv = nn.Conv1d(
                in_channels=in_ch, out_channels=in_ch,
                kernel_size=k, groups=in_ch, padding=k // 2, bias=False)
            self.pointwise_conv = nn.Conv1d(
                in_channels=in_ch, out_channels=out_ch,
                kernel_size=1, padding=0, bias=bias)
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(
                in_channels=in_ch, out_channels=in_ch,
                kernel_size=k, groups=in_ch, padding=k // 2, bias=False)
            self.pointwise_conv = nn.Conv2d(
                in_channels=in_ch, out_channels=out_ch,
                kernel_size=1, padding=0, bias=bias)
        else:
            raise Exception("Incorrect dimension!")

    def forward(self, x):
        """
        :Input: (batch_num, in_ch, seq_length)
        :Output: (batch_num, out_ch, seq_length)
        """
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))


class Initialized_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding=0, groups=1,
                 relu=False, bias=False):
        super().__init__()
        self.out = nn.Conv1d(
            in_channels, out_channels,
            kernel_size, stride=stride,
            padding=padding, groups=groups, bias=bias)
        if relu is True:
            self.relu = True
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            self.relu = False
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu is True:
            return F.relu(self.out(x))
        else:
            return self.out(x)
