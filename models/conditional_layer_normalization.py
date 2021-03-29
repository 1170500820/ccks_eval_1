import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalLayerNormalization(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, pass_layer_norm=False):
        """

        :param hidden_size:
        :param eps:
        :param pass_layer_norm: if True, just return the representation, pass the cln procedure
        """
        super(ConditionalLayerNormalization, self).__init__()
        self.weight_map = nn.Linear(hidden_size, hidden_size, bias=False)   # todo bias=False?
        self.bias_map = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_size = hidden_size
        self.eps = eps
        self.pass_layer_norm = pass_layer_norm

    def forward(self, representation, condition):
        """
        稍微修改了一下repr_mean和repr_var的计算方法，现在对于非batch的，即rep (seq_l, hidden), cond (1, hidden)的情况，也能正常算
        :param representation: (bsz, seq_l, hidden_size), sentence representation that does not contain <CLS> and <SEP>
        :param condition: (bsz, 1, hidden_size), pooled type embedding
        :return: denormed_repr (bsz, seq_l, hidden_size)
        """
        if self.pass_layer_norm:
            return representation
        weight = self.weight_map(condition) # (bsz, 1, hidden_size)
        bias = self.bias_map(condition) # (bsz, 1, hidden_size)
        # (bsz, 1, hidden) or (bsz, hidden)
        # normed_repr = F.layer_norm(representation, [self.hidden_size], weight, bias) # RuntimeError
        repr_mean = torch.mean(representation, dim=-2).unsqueeze(dim=-2)   # (bsz, 1, hidden)
        repr_var = torch.var(representation, dim=-2, unbiased=False).unsqueeze(dim=-2) # (bsz, 1, hidden)
        normed_repr = (representation - repr_mean) / torch.sqrt(repr_var + self.eps)    # (bsz, seq_l, hidden)
        denormed_repr = weight * normed_repr + bias # weight and bias (bsz, 1, hidden_size)
        # 这个CLN可以手动实现吗
        return denormed_repr
