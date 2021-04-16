import sys
sys.path.append('..')
import torch
from models.sentence_representation_layer import *
from models.trigger_extraction_model import *
from models.argument_extraction_model import *
from models.role_mask import *
import pickle


def output2spans(model_output, threshold=0.5):
    """
    将模型的输出转化为span的形式
    对应的是双linear，softmax的形式
    :param model_output: (bsz, seq_l, role_types) x 2
    :return:
    """
    starts, ends = model_output
    single = False
    if len(starts.size()) == 2:
        single = True
        starts = torch.unsqueeze(starts, dim=0)
        ends = torch.unsqueeze(ends, dim=0)
    bsz, seq_l, role_types = starts.size()
    # 转为(bsz, role_types, seq_l)
    starts, ends = starts.permute([0, 2, 1]), ends.permute([0, 2, 1])
    start_result, end_result = torch.gt(starts, threshold).long(), torch.gt(ends, threshold).long()

    # bsz, role type, token
    batch_spans = []
    for b in range(bsz):
        role_spans = []
        for r in range(role_types):
            spans = argument_span_determination(start_result[b][r], end_result[b][r], starts[b][r], ends[b][r])
            role_spans.append(spans)
        batch_spans.append(role_spans)
    if single:
        return batch_spans[0]
    else:
        return batch_spans


def argument_span_determination(binary_start: [], binary_end: [], prob_start: [], prob_end: []):
    """
    来自paper: Exploring Pre-trained Language Models for Event Extraction and Generation
    Algorithm 1
    :param binary_start:
    :param binary_end:
    :param prob_start:
    :param prob_end:
    :return:
    """
    a_s, a_e = -1, -1
    state = 1
    # state
    #   1 - 在外面
    #   2 - 遇到了一个start
    #   3 - 在start之后遇到了一个end，看看还有没有更长的end
    spans = []
    seq_l = len(binary_start)
    for i in range(seq_l):
        if state == 1 and binary_start[i] == 1:
            a_s = i
            state = 2
        elif state == 2:
            # 什么叫new start?
            if binary_start[i] == 1:
                if prob_start[i] > prob_start[a_s]:
                    a_s = i
            if binary_end[i] == 1:
                a_e = i
                state = 3
        elif state == 3:
            if binary_end[i] == 1:
                if prob_end[i] > prob_end[a_e]:
                    a_e = i
            if binary_start[i] == 1:
                spans.append([a_s, a_e])
                a_s, a_e = i, -1
                state = 2
    if state == 3:  # todo 这个debug有问题吗？
        spans.append([a_s, a_e])
    return spans


def load_model_ae_aem(path, n_head, hidden_size, d_head, hidden_dropout_prob, syntactic_feature_size):
    aem = ArgumentExtractionModel(n_head, hidden_size, d_head, hidden_dropout_prob, syntactic_feature_size)
    aem.load_state_dict(torch.load(path))
    aem.eval()
    return aem


def load_model_ae_repr(path, PLM_path, hidden_size):
    repr_model = SentenceRepresentation(PLM_path, hidden_size)
    repr_model.load_state_dict(torch.load(path))
    repr_model.eval()
    return repr_model


def load_model_ae_trigger_repr(path, hidden_size):
    trigger_repr_model = TriggeredSentenceRepresentation(hidden_size)
    trigger_repr_model.load_state_dict(torch.load(path))
    trigger_repr_model.eval()
    return trigger_repr_model


def load_model_ae_role_mask(rfief_path):
    role_mask = RoleMask(pickle.load(open(rfief_path, 'rb')))
    return role_mask


def count_components(lst):
    """
    对lst中的元素进行计数
    lst的元素必须是hashable
    :param lst:
    :return:
    """
    cnt = {}
    for a in lst:
        if a in cnt:
            cnt[a] = cnt[a] + 1
        else:
            cnt[a] = 1
    return cnt


def count_components_percentage(lst):
    """
    求出lst每个元素所占的百分比
    lst中的元素必须是hashable
    :param lst:
    :return:
    """
    cnt = count_components(lst)
    total = sum(map(lambda x: x[1], cnt.items()))
    per = {}
    for key, value in cnt.items():
        per[key] = value / total
    return per