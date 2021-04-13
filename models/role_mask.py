import torch
import torch.nn as nn
import torch.nn.functional as F
from settings import event_types_init, role_types, event_available_roles


class RoleMask(nn.Module):
    def __init__(self, rfief):
        super(RoleMask, self).__init__()
        self.rfief = rfief
        self.mask = {}
        self.weights = {}
        self.generate_mask()

    def generate_mask(self):
        """
        generate mask for each type
        generate weight for each type-role
        :return:
        """
        for t in event_types_init:
            cur_type_mask = torch.ones(len(role_types), dtype=torch.float)
            for i in range(len(role_types)):
                if role_types[i] not in event_available_roles[t]:
                    cur_type_mask[i] = 0
            self.mask[t] = cur_type_mask
        for t in event_types_init:
            cur_type_weight = torch.zeros(len(role_types), dtype=torch.float)
            for i in range(len(role_types)):
                cur_type_weight[i] = self.rfief[(t, role_types[i])]
            self.weights[t] = cur_type_weight

    def return_mask(self, logits, batch_event_types):
        seq_l = logits.size(1)
        return torch.stack(list(map(lambda x: self.mask[x].repeat(seq_l, 1), batch_event_types))).cuda()

    def return_weighted_mask(self, logits, batch_event_types):
        """
        读取模型输出的预测结果和batch中的事件类型列表，输出用于求BCEloss的权值矩阵
        mask是1/0矩阵
        weight是rfief的矩阵
        返回值是二者点积
        :param logits: 模型的预测结果 (bsz, seq_l, len(role_types))
        :param batch_event_types: 事件类型列表 (type_str1, type_str2, ...) len=bsz
        :return: (bsz, seq_l, len(role_types))
        """
        seq_l = logits.size(1)
        mask = self.return_mask(logits, batch_event_types)  # (bsz, seq_l, len(role_types))
        weight = torch.stack(list(map(lambda x: self.mask[x].repeat(seq_l, 1), batch_event_types))).cuda()  # (bsz, seq_l, len(role_types))
        return mask * weight    # (bsz, seq_l, len(role_types))

    def forward(self, logits, batch_event_types):
        """

        :param logits: (bsz, seq_l, len(roles))
        :param batch_event_types: [type1, type2, ...] of size bsz
        :return:
        """
        seq_l = logits.size(1)
        logits_mask = torch.stack(list(map(lambda x: self.mask[x].repeat(seq_l, 1), batch_event_types))).cuda()
        logits = logits * logits_mask
        return logits
