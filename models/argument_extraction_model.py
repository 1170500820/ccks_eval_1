import torch
import torch.nn as nn
import torch.nn.functional as F
from .self_attentions import SelfAttn


class ArgumentExtractionModel(nn.Module):
    def __init__(self, n_head, hidden_size, d_head, dropout_prob, syntactic_size, skip_attn=False, skip_syn=False, skip_RPE=False):
        """

        :param n_head:
        :param hidden_size:
        :param d_head:
        :param dropout_prob:
        :param syntactic_size:
        :param pass_attn:
        :param pass_syn:
        """
        super(ArgumentExtractionModel, self).__init__()

        self.n_head = n_head
        self.hidden_size = hidden_size
        self.d_head = d_head
        self.dropout_prob = dropout_prob
        self.syntactic_size = syntactic_size
        self.skip_attn = skip_attn
        self.skip_syn = skip_syn
        self.skip_RPE = skip_RPE

        self.self_attn = SelfAttn(self.n_head, self.d_head, self.hidden_size, self.dropout_prob)

        # FCN for trigger finding
        # origin + attn(origin) + syntactic + RPE
        self.fcn_start = nn.Linear(self.hidden_size * 2 + self.syntactic_size + 1, 1)
        self.fcn_end = nn.Linear(self.hidden_size * 2 + self.syntactic_size + 1, 1)

        self.init_weights()

    def forward(self, cln_embeds, syntactic_structure, relative_positional_encoding):
        """

        :param cln_embeds: (bsz, seq_l, hidden_size)
        :param syntactic_structure: (bsz, seq_l, syntactic_size)
        :param relative_positional_encoding: (bsz, seq_l, 1) todo 无效区域的距离设为inf还是0
        :return:
        """
        # self attention (multihead attention)
        attn_out = self.self_attn(cln_embeds)

        # concatenation
        if self.skip_attn:
            attn_out = torch.zeros(attn_out.size()).cuda()
        if self.skip_syn:
            syntactic_structure = torch.zeros(syntactic_structure.size()).cuda()
        if self.skip_RPE:
            relative_positional_encoding = torch.zeros(relative_positional_encoding.size()).cuda()
        final_repr = torch.cat((cln_embeds, attn_out, syntactic_structure, relative_positional_encoding), dim=-1)
