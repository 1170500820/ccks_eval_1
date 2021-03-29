import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # W_q, W_k and W_v
        self.q_net = nn.Linear(self.hidden_size, self.num_heads * self.d_head, bias=False)
        self.kv_net = nn.Linear(self.hidden_size, 2 * self.num_heads * self.d_head, bias=False)
        self.scale = 1 / self.d_head ** 0.5

        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_prob)

        # todo whats this?
        self.dropout_att = nn.Dropout(self.dropout_prob)

        # O matrix to combine O of multiple heads
        self.o_net = nn.Linear(self.num_heads * self.d_head, self.hidden_size, bias=False)

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
        pass

    def forward(self, cln_embeds, syntactic_structure):
        """

        :param cln_embeds: (bsz, seq_l, hidden_size)， 经过CLN处理的句子embeddings
        :param syntactic_structure: (bsz, seq_l, syntactic_size)， 将会拼接到embeds上
        :return:
        """
        # self attention (multihead attention)
        #   get q,k,v
        head_q = self.q_net(cln_embeds) # (bsz, seq_l, num_heads * d_head)
        head_k, head_v = torch.chunk(self.kv_net(cln_embeds), 2, -1)    # both (bsz, seq_l, num_heads * d_head)
        #   todo 这里是在干什么？
        #   turn q, k, v's sizes into (bsz, seq_l, num_heads, d_head)
        head_q = head_q.view(cln_embeds.size(0), cln_embeds.size(1), self.num_heads, self.d_head)
        head_k = head_k.view(cln_embeds.size(0), cln_embeds.size(1), self.num_heads, self.d_head)
        head_v = head_v.view(cln_embeds.size(0), cln_embeds.size(1), self.num_heads, self.d_head)
        attn_score = torch.einsum('bind,bjnd->ijbn', head_q, head_k)    # got (seq_l_i, seq_l_j, bsz, num_heads)
        attn_score.mul_(self.scale)
        #   softmax on j dimension
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropout_att(attn_prob) # 可选的dropout？
        #   O cal
        attn_vec = torch.einsum('ijbn,bjnd->bind', attn_prob, head_v)   # got (bsz, seq_l_i, num_heads, d_head)
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.num_heads * self.d_head)
        #   final projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.dropout(attn_out)   # 有一个dropout

        # concatenation
        # print('cln',  cln_embeds.size())
        # print('att', attn_out.size())
        # print('syn', syntactic_structure.size())
        final_repr = torch.cat((cln_embeds, attn_out, syntactic_structure.cuda()), dim=-1) # got (bsz, seq_l, hidden * 2 + syn)
        # linear
        #   got both (bsz, seq_l, 1), convert to (bsz, seq_l)
        start_logits, end_logits = self.fcn_start(final_repr).squeeze(), self.fcn_end(final_repr).squeeze()
        # sigmoid
        # starts, ends = F.sigmoid(start_logits), F.sigmoid(end_logits)   # got both (bsz, seq_l, 2)
        # todo 暂时跳过，使用binary_cross_entropy_with_logits
        starts, ends = F.softmax(start_logits), F.softmax(end_logits)   # both (bsz, seq_l)
        return starts, ends
