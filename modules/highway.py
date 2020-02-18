"""
Highway network.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn import Initialized_Conv1d


class Highway_old(nn.Module):
    """
    Applies highway transformation to the incoming data.
    It is like LSTM that uses gates. Highway network is
    helpful to train very deep neural networks.
    y = H(x, W_H) * T(x, W_T) + x * C(x, W_C)
    C = 1 - T
    :Examples:
        >>> m = Highway(2, 300)
        >>> x = torch.randn(32, 20, 300)
        >>> y = m(x)
        >>> print(y.size())
    """

    def __init__(self, layer_num, size):
        """
        :param layer_num: number of highway transform layers
        :param size: size of the last dimension of input
        """
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList(
            [nn.Linear(size, size) for _ in range(self.n)])
        self.gate = nn.ModuleList(
            [nn.Linear(size, size) for _ in range(self.n)])
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        :Input: (N, *, size) * means any number of additional dimensions.
        :Output: (N, *, size) * means any number of additional dimensions.
        """
        for i in range(self.n):
            gate = F.sigmoid(self.gate[i](x))
            nonlinear = self.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        return x


# TODO: compare the difference between two highway
class Highway(nn.Module):
    def __init__(self, layer_num, size):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList(
            [Initialized_Conv1d(size, size, relu=False, bias=True)
             for _ in range(self.n)])
        self.gate = nn.ModuleList(
            [Initialized_Conv1d(size, size, relu=False, bias=True)
             for _ in range(self.n)])

    def forward(self, x):
        # x: shape [batch_size, hidden_size, length]
        dropout = 0.1
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = self.linear[i](x)
            nonlinear = F.dropout(nonlinear, p=dropout, training=self.training)
            x = gate * nonlinear + (1 - gate) * x
        return x
