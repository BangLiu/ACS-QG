"""
Initialize the decoder's initial state.
"""
import torch
import torch.nn as nn


class DecIniter(nn.Module):
    """
    Use encoder last hidden state to initialize the decoder's initial state.
    """

    def __init__(self, config):
        """
        Initialize the Linear and Tanh layer for initializer.
        """
        super(DecIniter, self).__init__()
        self.num_directions = 2 if config.brnn else 1
        assert config.enc_rnn_size % self.num_directions == 0
        self.enc_rnn_size = config.enc_rnn_size
        self.dec_rnn_size = config.dec_rnn_size
        # initialize decoder hidden state
        input_dim = self.enc_rnn_size // self.num_directions
        if config.use_style_info:
            input_dim += config.style_emb_dim
        self.initer = nn.Linear(
            input_dim,
            self.dec_rnn_size)
        self.tanh = nn.Tanh()

    def forward(self, enc_list):
        """
        Use encoder's last backward hidden state as input,
        project to decoder rnn size by linear + tanh layers.
        If we have more encoding, such as style, we will cat them.
        NOTE: last backward hidden state remembers more clear about
        head input words, which are more import to head output words.
        If we use single layer RNN for encoder, usually the input word order
        is also reverse due to this reason.
        :param enc_list: a list of tensors used for initializer.
            Input shape - ......
        :return: the initial value for decoder hidden state.
        TODO: revise code to make input and output shape be [batch, length, dim]
        """
        x = torch.cat(enc_list, dim=1)  # NOTICE: dimension
        return self.tanh(self.initer(x))
