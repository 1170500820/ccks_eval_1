import torch
import torch.nn as nn
import torch.nn.functional as F
from settings import event_types_init, role_types, event_available_roles


class RoleMask(nn.Module):
    def __init__(self):
        super(RoleMask, self).__init__()
        self.mask = {}
        self.generate_mask()

    def generate_mask(self):
        """
        generate mask for each type
        :return:
        """
        for t in event_types_init:
            cur_type_mask = torch.ones(len(role_types), dtype=torch.float)
            for i in range(len(role_types)):
                if role_types[i] not in event_available_roles[t]:
                    cur_type_mask[i] = 0
            self.mask[t] = cur_type_mask

    def return_mask(self, logits, batch_event_types):
        seq_l = logits.size(1)
        return torch.stack(list(map(lambda x: self.mask[x].repeat(seq_l, 1), batch_event_types))).cuda()

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
