"""
Implement input sentence encoder.
"""
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from .config import *
from common.constants import DEVICE
from util.tensor_utils import to_sorted_tensor, to_original_tensor


class Encoder(nn.Module):
    """
    Transform embeddings to encoding representations.
    """

    def __init__(self, config, input_size, dropout=0.1):
        """
        Initialize a GRU encoder.
        :param config: configuration, includes total enc size, is bi-direction, etc.
        :param input_size: input dimension.
        :param dropout: dropout rate for GRU
        """
        super(Encoder, self).__init__()
        self.config = config
        self.layers = config.layers
        self.num_directions = 2 if config.brnn else 1
        assert config.enc_rnn_size % self.num_directions == 0
        self.hidden_size = config.enc_rnn_size // self.num_directions
        self.rnn = nn.GRU(
            input_size, self.hidden_size,
            num_layers=config.layers, dropout=config.dropout,
            bidirectional=config.brnn, batch_first=True)

    def forward(self, input_emb, lengths, hidden=None):
        """
        Given input embeddings and input seq lengths, calculate encoding representations.
        :param input_emb: embedding of a batch.
            Input shape - [seq_len, batch_size, hidden_dim]
        :param lengths: lengths of each sample.
        :param hidden: hidden of previous layer. Default None.
        :return: encoding of a batch.
            Output shape - [unpadded_max_thisbatch_seq_len, batch_size, hidden_dim * num_layers]
        TODO: revise code to make input and output shape be [batch, length, dim]
        """
        # input_emb shape: [seq_len, batch_size, hidden_dim] [100, 32, 412]
        # sorted_emb shape: [seq_len, batch_size, hidden_dim] [100, 32, 412]
        sorted_input_emb, sorted_lengths, sorted_idx = to_sorted_tensor(
            input_emb, lengths, sort_dim=1, device=DEVICE)
        emb = pack(sorted_input_emb, sorted_lengths, batch_first=False)
        self.rnn.flatten_parameters()
        outputs, hidden_t = self.rnn(emb, hidden)
        # hidden_t shape:  [num_layers, batch_size, hidden_dim] [2, 32, 256]
        # outputs shape: [unpadded_seq_len, batch_size, hidden_dim * num_layers] [79, 32, 512]
        # !!! NOTICE: it will unpack to max_unpadded_length.
        outputs = unpack(outputs, batch_first=False)[0]
        outputs = to_original_tensor(
            outputs, sorted_idx, sort_dim=1, device=DEVICE)
        return hidden_t, outputs
