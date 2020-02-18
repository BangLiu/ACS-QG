import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class ScaledDotProductAttention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention'
    """

    def __init__(self, dropout=0.0):
        """
        :param dropout: attention dropout rate
        """
        super().__init__()
        self.dropout = dropout

    def forward(self, query, key, value, mask=None):
        """
        :param query: (batch_num, query_length, d_model)
        :param key: (batch_num, key_length, d_model)
        :param value: (batch_num, key_length, d_model)
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill_(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = F.dropout(p_attn, p=self.dropout)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    """
    Compute 'Multi-Head Attention'
    When we calculate attentions, usually key and value are the same tensor.
    For self-attention, query, key, value are all the same tensor.
    """

    def __init__(self, h, d_model, dropout=0.1):
        """
        :param h: number of heads
        :param d_model: hidden size
        :param dropout: attention dropout rate
        """
        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attention = ScaledDotProductAttention(dropout)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        """
        :param query: (batch_num, query_length, d_model)
        :param key: (batch_num, key_length, d_model)
        :param value: (batch_num, key_length, d_model)
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))]

        # Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(
            query, key, value, mask=mask)

        # "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(
            nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x), self.attn


class ConcatAttention(nn.Module):
    def __init__(self, context_dim, query_dim, att_dim):
        super(ConcatAttention, self).__init__()
        self.context_dim = context_dim
        self.query_dim = query_dim
        self.att_dim = att_dim
        self.linear_pre = nn.Linear(context_dim, att_dim, bias=True)
        self.linear_q = nn.Linear(query_dim, att_dim, bias=False)
        self.linear_v = nn.Linear(att_dim, 1, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context, precompute=None):
        """
        input: batch x query_dim
        context: batch x sourceL x context_dim
        """
        # pre-compute projected context, so that it don't
        # need to be calculated multiple times.
        if precompute is None:
            precompute00 = self.linear_pre(
                context.contiguous().view(-1, context.size(2)))
            # batch x seqL x att_dim
            precompute = precompute00.view(
                context.size(0), context.size(1), -1)
        # batch x 1 x att_dim
        targetT = self.linear_q(input).unsqueeze(1)
        # batch x seqL x att_dim
        tmp10 = precompute + targetT.expand_as(precompute)
        # batch x sourceL x att_dim
        tmp20 = self.tanh(tmp10)
        # batch x sourceL
        energy = self.linear_v(
            tmp20.view(-1, tmp20.size(2))).view(tmp20.size(0), tmp20.size(1))
        if self.mask is not None:
            energy = energy * (1 - self.mask) + self.mask * (-1000000)
        # batch x sourceL
        score = self.sm(energy)
        # batch x 1 x sourceL
        score_m = score.view(score.size(0), 1, score.size(1))
        # batch x context_dim
        weightedContext = torch.bmm(score_m, context).squeeze(1)

        return weightedContext, score, precompute


if __name__ == "__main__":
    # Test multi-head attention
    n_head = 8
    d_model = 128
    d_k = d_model // n_head
    d_v = d_model // n_head
    batch_num = 10
    len_q = 20
    len_k = 30
    q = torch.rand(batch_num, len_q, d_model)
    k = torch.rand(batch_num, len_k, d_model)
    v = k
    model = MultiHeadAttention(n_head, d_model, dropout=0.1)
    output, attn = model(q, k, v)
    print(output.shape)
    print(attn.shape)
