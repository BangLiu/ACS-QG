import torch.nn as nn


class MaxOut(nn.Module):
    """
    Given a tensor, apply max pooling to its last dim
    according to a specific pool size.
    """
    def __init__(self, pool_size):
        super(MaxOut, self).__init__()
        self.pool_size = pool_size

    def forward(self, input):
        input_size = list(input.size())
        assert input_size[-1] % self.pool_size == 0
        output_size = [d for d in input_size]
        output_size[-1] = output_size[-1] // self.pool_size
        output_size.append(self.pool_size)
        last_dim = len(output_size) - 1
        input = input.view(*output_size)
        input, idx = input.max(last_dim, keepdim=True)
        output = input.squeeze(last_dim)

        return output
