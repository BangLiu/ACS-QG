"""
Generate output vocabulary prob distribution
based on decoder final output vector.
"""
import torch.nn as nn


class Generator(nn.Module):
    """
    Use decoder's final output vector to generate output word
    probabilities.
    """
    def __init__(self, input_size, predict_size):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(input_size, predict_size),
            nn.Softmax(dim=1))

    def forward(self, g_output_t):
        return self.generator(g_output_t)
