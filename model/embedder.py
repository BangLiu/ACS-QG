import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import *
from common.constants import DEVICE


class Embedder(nn.Module):
    """
    Embedding different features according to configuration
    and concatenate all the embeddings.
    """

    def __init__(self, config, emb_mats, emb_dicts, dropout=0.1):
        """
        Initialize two dicts: self.embs and self.conv2ds.
        self.embs contains different Embedding layers for different tags.
        self.conv2ds contains different Conv2d layers for different tags
        that requires convolution, e.g., character or BPE embedding.
        :param config: arguments
        :param emb_mats: dict of embedding matrices for different tags
        :param emb_dicts: dict of word2id dicts for different tags.
        :param dropout: dropout rate for dropout layers after embedding and after convolution
        """
        super().__init__()
        self.config = config
        self.embs = torch.nn.ModuleDict()
        self.conv2ds = torch.nn.ModuleDict()
        # construct all keys, so we reuse one embedder
        # and can train on different tasks
        # for tag in emb_mats.keys():
        for tag in config.emb_tags:
            if config.emb_config[tag]["need_emb"]:
                self.embs.update(
                    {tag:
                     nn.Embedding.from_pretrained(
                         torch.FloatTensor(emb_mats[tag]),
                         freeze=(not config.emb_config[tag]["trainable"]))})
                if config.emb_config[tag]["need_conv"]:
                    self.conv2ds.update(
                        {tag:
                         nn.Conv2d(
                             config.emb_config[tag]["emb_dim"], config.d_model,
                             kernel_size=(1, 5), padding=0, bias=True)})
                    nn.init.kaiming_normal_(
                        self.conv2ds[tag].weight, nonlinearity='relu')

        # self.conv1d = Initialized_Conv1d(
        #     total_emb_dim, config.d_model, bias=False)
        # self.high = Highway(2, config.d_model)
        self.dropout = dropout

    def get_total_emb_dim(self, emb_tags):
        """
        Given tags to embed, get the total dimension of embeddings according to configure.
        :param emb_tags: a list of tags to indicate which tags we will embed as input
        :return: total embedding dimension
        """
        total_emb_dim = 0
        for tag in emb_tags:
            if self.config.emb_config[tag]["need_emb"]:
                if self.config.emb_config[tag]["need_conv"]:
                    total_emb_dim += self.config.d_model
                else:
                    total_emb_dim += self.config.emb_config[tag]["emb_dim"]
            else:
                total_emb_dim += 1  # use feature value itself as embedding
        return total_emb_dim

    def forward(self, batch, field, emb_tags):
        """
        Given a batch of data, the field and tags we want to emb,
        return the concatenated embedding representation.
        :param batch: a batch of data. It is a dict of tensors.
            Each tensor is tag ids or tag values.
            Input shape - [batch_size, seq_length]
        :param field: which field we want to embed.
            For example, "ques", "ans_sent", etc.
        :param emb_tags: a list of tags to indicate which tags we will embed
        :return: concatenated embedding representation
            Output shape - [batch_size, emb_dim, seq_length]
        TODO: revise code to make input and output shape be [batch, length, dim]
        """
        emb = torch.FloatTensor().to(DEVICE)
        # use emb_tags to control which tags are actually in use
        for tag in emb_tags:
            field_id = field + "_" + tag + "_ids"  # NOTICE: naming style is same with data loader
            field_tag = field + "_" + tag
            if self.config.emb_config[tag]["need_emb"]:
                tag_emb = self.embs[tag](batch[field_id])
            else:
                tag_emb = batch[field_tag].unsqueeze(2)
            if self.config.emb_config[tag]["need_conv"]:
                tag_emb = tag_emb.permute(0, 3, 1, 2)
                tag_emb = F.dropout(
                    tag_emb, p=self.dropout, training=self.training)
                tag_emb = self.conv2ds[tag](tag_emb)
                tag_emb = F.relu(tag_emb)
                tag_emb, _ = torch.max(tag_emb, dim=3)
            else:
                tag_emb = F.dropout(
                    tag_emb, p=self.dropout, training=self.training)
                tag_emb = tag_emb.transpose(1, 2)
            emb = torch.cat([emb, tag_emb], dim=1)
        # emb = self.conv1d(emb)
        # emb = self.high(emb)
        return emb
