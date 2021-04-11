import torch
import torch.nn as nn
import torch.nn.functional as F
from .self_attentions import SelfAttn


class TriggerExtractionModel(nn.Module):
    def __init__(self, num_heads, hidden_size, d_head, dropout_prob, syntactic_size, pass_attn=False, pass_syn=False):
        """

        :param num_heads:
        :param hidden_size:
        :param d_head:
        :param dropout_prob:
        :param syntactic_size:
        :param pass_attn: if True, pass the self attention procedure, return the original cln embeds
        :param pass_syn: if True, remove the syntactic feature, must be synchronized with settings.py
        """
        super(TriggerExtractionModel, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.d_head = d_head
        self.dropout_prob = dropout_prob
        self.syntactic_size = syntactic_size
        self.pass_attn = pass_attn
        self.pass_syn = pass_syn

        self.self_attn = SelfAttn(self.num_heads, self.d_head, self.hidden_size, self.dropout_prob)

        # FCN for trigger finding
        self.fcn_start = nn.Linear(self.hidden_size * 2 + self.syntactic_size, 1)
        self.fcn_end = nn.Linear(self.hidden_size * 2 + self.syntactic_size, 1)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.fcn_start.weight)
        torch.nn.init.xavier_uniform(self.fcn_end.weight)
        self.fcn_start.bias.data.fill_(0)
        self.fcn_end.bias.data.fill_(0)
        torch.nn.init.xavier_uniform(self.q_net.weight)
        torch.nn.init.xavier_uniform(self.kv_net.weight)
        torch.nn.init.xavier_uniform(self.o_net.weight)

    def forward(self, cln_embeds, syntactic_structure):
        """

        :param cln_embeds: (bsz, seq_l, hidden_size)， 经过CLN处理的句子embeddings
        :param syntactic_structure: (bsz, seq_l, syntactic_size)， 将会拼接到embeds上
        :return:
        """
        # self attention (multihead attention)
        attn_out = self.self_attn(cln_embeds)

        # concatenation
        # print('cln',  cln_embeds.size())
        # print('att', attn_out.size())
        # print('syn', syntactic_structure.size())
        if self.pass_attn:
            attn_out = torch.zeros(attn_out.size()).cuda()
        if self.pass_syn:
            syntactic_structure = torch.zeros(syntactic_structure.size())
        final_repr = torch.cat((cln_embeds, attn_out, syntactic_structure.cuda()), dim=-1) # got (bsz, seq_l, hidden * 2 + syn)
        # linear
        #   got both (bsz, seq_l, 1), convert to (bsz, seq_l)
        start_logits, end_logits = self.fcn_start(final_repr).squeeze(), self.fcn_end(final_repr).squeeze()
        # sigmoid
        starts, ends = F.sigmoid(start_logits), F.sigmoid(end_logits)   # got both (bsz, seq_l)
        # todo 暂时跳过，使用binary_cross_entropy_with_logits
        # starts, ends = F.softmax(start_logits), F.softmax(end_logits)   # both (bsz, seq_l)
        return starts, ends
