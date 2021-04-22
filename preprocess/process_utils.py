import torch
import torch.nn.functional as F
from settings import label_smoothing_range
import random


def randomize(*lsts):
    """
    randomize([1,2,3], [4,5,6])
    打乱顺序
    :param lsts: iterables of iterables,
    :return:
    """
    zipped = list(zip(*lsts))
    random.shuffle(zipped)
    return list(zip(*zipped))


def train_val_split(lst, split_ratio=None):
    cnt = len(lst)
    bound = int(cnt * split_ratio)
    return lst[:bound], lst[bound: ]


def multi_train_val_split(*lst, split_ratio=None):
    """

    :param lst: [lst1, lst2, ...]
    :param split_ratio:
    :return: train_lst1
    """
    trains, vals = [], []
    for l in lst:
        train_l, val_l = train_val_split(l, split_ratio=split_ratio)
        trains.append(train_l)
        vals.append(val_l)
    return trains + vals


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


def batchify_tensor(tensors_lst: [], bsz=None, pad=0, keep_tail=False):
    """

    :param tensors_lst: [tensor1, tensor2, ...]. Tensor shape: (L x *)
    :param bsz:
    :param pad:
    :param keep_tail: if True, keep the last batch whose batch size might be lower than bsz
    :return: [tensor_batch1, tensor_batch2, ...]. Tensor batch shape: (bsz, max(L), *)
    """
    temp_tensors, tensor_batches = [], []
    for i, current_tensor in enumerate(tensors_lst):
        temp_tensors.append(tensors_lst[i])
        if len(temp_tensors) == bsz:
            batched_tensor = torch.nn.utils.rnn.pad_sequence(temp_tensors, batch_first=True, padding_value=pad)
            tensor_batches.append(batched_tensor)
            temp_tensors = []
    if keep_tail and len(temp_tensors) != 0:
        tensor_batches.append(torch.nn.utils.rnn.pad_sequence(temp_tensors, batch_first=True, padding_value=pad))
    return tensor_batches


def batchify_iterable(lst: [], bsz=None, keep_tail=False):
    """

    :param lst: [v1, v2, ...]
    :param bsz:
    :param keep_tail:
    :return: [[v1, v2, ..., v_bsz], ...]
    """
    temp_lst, lst_batches = [], []
    for i, v in enumerate(lst):
        temp_lst.append(v)
        if len(temp_lst) == bsz:
            lst_batches.append(temp_lst)
            temp_lst = []
    if keep_tail and len(temp_lst) != 0:
        lst_batches.append(temp_lst)
    return lst_batches


def batchify_dict_of_tensors(lst: [dict, ], bsz=None, keep_tail=False):
    """
    返回的仍然是dict的list
    每个dict的tensor现在是输入的bsz个dict中tensor的batch化
    :param lst: [dict1, dict2, ...]. dict : {key1: tensor1, key2, tensor2, ...}, every dict should have same keys
    :param bsz:
    :param keep_tail:
    :return: [batched_dict1, batched_dict2, ...]. batched_dict: {key1: batched_tensor1, key2: batched_tensor2, ...}
    """
    dict_tensors = {}
    for d in lst:
        for key, value in d.items():
            if key not in dict_tensors:
                dict_tensors[key] = []
            dict_tensors[key].append(value.squeeze(dim=0))
    dict_batched_tensors = {}
    batch_cnt = 0
    for key, value in dict_tensors.items():
        dict_batched_tensors[key] = batchify_tensor(value, bsz=bsz, keep_tail=keep_tail)
        batch_cnt = len(dict_batched_tensors[key])
    result_dicts = []
    for i in range(batch_cnt):
        cur_dict = {}
        for key, value in dict_batched_tensors.items():
            cur_dict[key] = value[i]
        result_dicts.append(cur_dict)
    return result_dicts


def batchify(*lsts, bsz=None, lst_types=None):
    """

    :param bsz:
    :param lsts: list of list.
    :param lst_types: list of batchify types, in {'tensor', 'iterable'， ‘dict_tensor}.
        tensors need paddings, iterables do not
        if not provided, all iterables in default
    :return: list of batchified list
    """
    if lst_types is None:
        lst_types = ['iterable'] * len(lsts)
    function_map = {
        'iterable': batchify_iterable,
        'tensor': batchify_tensor,
        'dict_tensor': batchify_dict_of_tensors,

        # in case i messed up
        'iterables': batchify_iterable,
        'tensors': batchify_tensor,
        'dict_tensors': batchify_dict_of_tensors
    }
    results = list(map(lambda x: function_map[x[1]](lsts[x[0]], bsz=bsz), enumerate(lst_types)))
    return results


if __name__ == '__main__':
    a1 = list(range(20))
    a2 = list(range(20, 40))
    results = batchify(a1, a2, bsz=4)
