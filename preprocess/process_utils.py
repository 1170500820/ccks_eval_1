import torch
import torch.nn.functional as F
from settings import label_smoothing_range


def label_smoothing(label_batch, smooth_range=label_smoothing_range):
    """
    label_batch: (bsz, seq_l)
    smooth_range:
    original label变为0.9, 其它均为0.1
    :param label_batch:
    :param smooth_range:
    :return:
    """
    label_copy = label_batch.clone().detach().cuda()
    label_batch = label_batch * 0.9
    bsz = label_batch.size(0)
    expanded_label = torch.cat((torch.zeros(bsz, smooth_range).cuda(), label_copy, torch.zeros(bsz, smooth_range).cuda()), dim=-1).cuda()
    for k in range(smooth_range):
        right_label = expanded_label.T[(smooth_range - k - 1): -(smooth_range + k + 1)].T * 0.1
        left_label = expanded_label.T[(smooth_range + k + 1): -(smooth_range - k - 1) if -(smooth_range - k - 1) != 0 else None].T * 0.1
        label_batch = label_batch + right_label + left_label
    return label_batch


def label_smoothing_multi(labels_batch, smooth_range=label_smoothing_range):
    """
    label_batch: (bsz, seq_l, num_classes)
    smooth_range:
    original label变为0.9, 其它均为0.1
    :param label_batch:
    :param smooth_range:
    :return:
    """
    label_copy = labels_batch.clone().detach().cuda()  # (bsz, seq_l, num_classes)
    label_batch = labels_batch * 0.9    # (bsz, seq_l, num_classes)
    bsz, seq_l, num_classes = label_batch.size()
    expanded_label = torch.cat((torch.zeros(bsz, num_classes, smooth_range).cuda(),
                                label_copy.permute([0, 2, 1]).cuda(),
                                torch.zeros(bsz, num_classes, smooth_range).cuda()), dim=-1).cuda()    # (bsz, num_classes, seq_l + 2 * smooth_range)
    for k in range(smooth_range):
        right_label = expanded_label.permute([2, 0, 1])[(smooth_range - k - 1): -(smooth_range + k + 1)].permute([1, 0, 2]) * 0.1   # (bsz, seq_l, num_classes)
        left_label = expanded_label.permute([2, 0, 1])[(smooth_range + k + 1): -(smooth_range - k - 1) if -(smooth_range - k - 1) != 0 else None].permute([1, 0, 2]) * 0.1  # (bsz, seq_l, num_classes)
        label_batch = label_batch + right_label + left_label
    return label_batch

