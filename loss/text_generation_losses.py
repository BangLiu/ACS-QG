# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


def NMTLoss(vocabSize):
    """
    Use NLLLoss as loss function of generator.
    Set weight of PAD as zero.
    """
    PAD_idx = 0
    weight = torch.ones(vocabSize)
    weight[PAD_idx] = 0
    crit = nn.NLLLoss(weight, reduction="sum")
    if torch.cuda.is_available():
        crit.cuda()
    return crit


def QGLoss(g_prob_t, g_targets,
           c_outputs, c_switch,
           c_gate_values, c_targets,
           crit, copyCrit):
    # loss func with copy mechanism
    c_output_prob = c_outputs * c_gate_values.expand_as(c_outputs) + 1e-8
    g_output_prob = g_prob_t * (1 - c_gate_values).expand_as(g_prob_t) + 1e-8

    c_output_prob_log = torch.log(c_output_prob)
    g_output_prob_log = torch.log(g_output_prob)

    c_output_prob_log = c_output_prob_log * \
        (c_switch.unsqueeze(2).expand_as(c_output_prob_log))
    g_output_prob_log = g_output_prob_log * \
        ((1 - c_switch).unsqueeze(2).expand_as(g_output_prob_log))

    g_output_prob_log = g_output_prob_log.view(-1, g_output_prob_log.size(2))
    c_output_prob_log = c_output_prob_log.view(-1, c_output_prob_log.size(2))
    # NOTICE !!!!! we can change how the loss is calculated.

    g_loss = crit(g_output_prob_log, g_targets.contiguous().view(-1))
    c_loss = copyCrit(c_output_prob_log, c_targets.contiguous().view(-1))
    total_loss = g_loss + c_loss
    return total_loss


def SoftQGLoss1(g_prob_t, g_targets,  # TODO
                c_outputs, c_switch,
                c_gate_values, c_targets,
                crit, copyCrit,
                soft_g_outputs, new_g_outputs):
    """
    In this loss func, we just mimic the original QGLoss.
    Soft copy is just used to revise the g_prob_t and get new_g_outputs.
    """
    # loss func with copy mechanism
    # print("c_gate_values shape ", c_gate_values.shape)  # [max_batch_output_seq_len, batch_size, 3]
    c_gate_values = c_gate_values[:, :, 0].unsqueeze(2)  # 0 column is copy, 1 column is generate, 2 column is soft-copy
    # print("c_gate_values shape ", c_gate_values.shape)  # [max_batch_output_seq_len, batch_size, 1]
    # print("c_outputs shape ", c_outputs.shape)  # [max_batch_output_seq_len, batch_size, max_batch_input_seq_len]
    g_prob_t = new_g_outputs

    c_output_prob = c_outputs * c_gate_values.expand_as(c_outputs) + 1e-8
    g_output_prob = g_prob_t * (1 - c_gate_values).expand_as(g_prob_t) + 1e-8

    c_output_prob_log = torch.log(c_output_prob)
    g_output_prob_log = torch.log(g_output_prob)

    c_output_prob_log = c_output_prob_log * \
        (c_switch.unsqueeze(2).expand_as(c_output_prob_log))
    g_output_prob_log = g_output_prob_log * \
        ((1 - c_switch).unsqueeze(2).expand_as(g_output_prob_log))

    g_output_prob_log = g_output_prob_log.view(-1, g_output_prob_log.size(2))
    c_output_prob_log = c_output_prob_log.view(-1, c_output_prob_log.size(2))
    # NOTICE !!!!! we can change how the loss is calculated.

    g_loss = crit(g_output_prob_log, g_targets.contiguous().view(-1))
    c_loss = copyCrit(c_output_prob_log, c_targets.contiguous().view(-1))
    total_loss = g_loss + c_loss
    return total_loss


def SoftQGLoss2(g_prob_t, g_targets,# TODO
                c_outputs, c_switch, c_switch_soft,
                c_gate_values, c_targets, c_targets_soft,
                crit, copyCrit,
                soft_g_outputs, new_g_outputs):
    """
    In this loss func, we also provide supervision for soft-copy outputs.
    """
    # loss func with copy mechanism
    c_output_prob = c_outputs * c_gate_values[:, :, 0].unsqueeze(2).expand_as(c_outputs) + 1e-8
    g_output_prob = g_prob_t * c_gate_values[:, :, 1].unsqueeze(2).expand_as(g_prob_t) + 1e-8
    soft_c_output_prob = soft_g_outputs * c_gate_values[:, :, 2].unsqueeze(2).expand_as(g_prob_t) + 1e-8

    c_output_prob_log = torch.log(c_output_prob)
    g_output_prob_log = torch.log(g_output_prob)
    soft_c_output_prob_log = torch.log(soft_c_output_prob)

    c_output_prob_log = c_output_prob_log * \
        (c_switch.unsqueeze(2).expand_as(c_output_prob_log))
    g_output_prob_log = g_output_prob_log * \
        ((1 - c_switch - c_switch_soft).unsqueeze(2).expand_as(g_output_prob_log))
    soft_c_output_prob_log = soft_c_output_prob_log * \
        (c_switch_soft.unsqueeze(2).expand_as(soft_c_output_prob_log))

    g_output_prob_log = g_output_prob_log.view(-1, g_output_prob_log.size(2))
    c_output_prob_log = c_output_prob_log.view(-1, c_output_prob_log.size(2))
    soft_c_output_prob_log = soft_c_output_prob_log.view(-1, soft_c_output_prob_log.size(2))
    # NOTICE !!!!! we can change how the loss is calculated.

    g_loss = crit(g_output_prob_log, g_targets.contiguous().view(-1))
    c_loss = copyCrit(c_output_prob_log, c_targets.contiguous().view(-1))
    soft_c_loss = crit(soft_c_output_prob_log, c_targets_soft.contiguous().view(-1))
    total_loss = g_loss + c_loss + soft_c_loss
    return total_loss
