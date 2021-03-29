import torch


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
