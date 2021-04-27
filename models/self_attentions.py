import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttn(nn.Module):
    def __init__(self, n_head, d_head, hidden_size, dropout_prob):
        super(SelfAttn, self).__init__()
        self.n_head = n_head
        self.d_head = d_head
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob

        # W_q, W_k, W_v
        self.q_net = nn.Linear(self.hidden_size, self.n_head * self.d_head, bias=False)
        self.kv_net = nn.Linear(self.hidden_size, 2 * self.n_head * self.d_head, bias=False)
        self.scale = 1 / self.d_head ** 0.5

        # Dropout Layer
        self.dropout = nn.Dropout(self.dropout_prob)
        self.dropout_attn = nn.Dropout(self.dropout_prob)

        # O matrix to combine O of multiple heads
        self.o_net = nn.Linear(self.n_head * self.d_head, self.hidden_size, bias=False)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.q_net.weight)
        torch.nn.init.xavier_uniform_(self.kv_net.weight)
        torch.nn.init.xavier_uniform_(self.o_net.weight)

    def forward(self, embeds):
        """

        :param embeds: (bsz, seq_l, hidden_size)
        :return: (bsz, seq_l, hidden_size) attentioned embeds
        """
        head_q = self.q_net(embeds) # (bsz, seq_l, n_head * d_head)
        head_k, head_v = torch.chunk(self.kv_net(embeds), 2, -1)    # (bsz, seq_l, n_head * d_head * 2)

        head_q = head_q.view(embeds.size(0), embeds.size(1), self.n_head, self.d_head)
        head_k = head_k.view(embeds.size(0), embeds.size(1), self.n_head, self.d_head)
        head_v = head_v.view(embeds.size(0), embeds.size(1), self.n_head, self.d_head)  # (bsz, seq_l, n_head, d_head)

        attn_score = torch.einsum('bind,bjnd->ijbn', head_q, head_k)    # (seq_l, seq_l, bsz, n_head)
        attn_score.mul_(self.scale)

        # softmax on j dimension to calculate attn weight on each seq token
        attn_prob = F.softmax(attn_score, dim=1)    # (seq_l, bsz, n_head)
        attn_prob = self.dropout_attn(attn_prob)    # (seq_l, bsz, n_head)

        # O
        attn_vec = torch.einsum('ijbn,bjnd->bind', attn_prob, head_v)  # (bsz, seq_l, n_head, d_head)
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # final projection
        attn_out = self.o_net(attn_vec) # (bsz, seq_l, hidden)
        attn_out = self.dropout(attn_out)   # (bsz, seq_l, hidden)

        return attn_out
