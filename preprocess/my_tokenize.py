"""
与tokenize相关的一些函数
"""
from transformers import BertTokenizer
from reader import *
from settings import *
import pickle


# 下面两个方法用于处理tokenize后的坐标差异问题
def generate_trigger_token(events: [], tokenized_ids: [], tokenizer: BertTokenizer):
    """
        为trigger词建立token标注序列

        使用bert对中文进行tokenize,得到的是字序列。
        该方法能够根据span生成对应的标注结果
        如果每个span对应的tag只有一个，则直接标注（默认加I-）
        如果有两个，则采用BI标注法
        不能大于两个
        默认是部分包含也算做tag
    :param tokenizer: tokenizer
    :param events: CCKS_Event类型，
    :param tokenized_ids: 只包含tokenize后的id, [[int, ], ]类型
    :return: trigger_taggers
    """
    trigger_taggers = []
    for i, e in enumerate(events):
        ids = tokenized_ids[i]
        content = e.content
        tokenized_seq = tokenizer.convert_ids_to_tokens(ids)
        taggers = ['O'] * len(tokenized_seq)
        for k in range(e.events_count):
            start, end, tag = e.events[k].trigger_start, e.events[k].trigger_end, e.events[k].event_type
            reach = 0
            reached_right, tagged_first = False, False  # 是否到达span右侧, 是否已标记第一个词
            unk_start_record = -1
            unk_last = False
            offset = 0
            for j, token in enumerate(tokenized_seq):
                ltoken = len(token)
                if token == '[UNK]':
                    if unk_last:
                        continue
                    else:
                        unk_start_record = reach
                        unk_last = True
                        continue
                elif unk_last:
                    end_ids = content.find(token, unk_start_record)
                    reach = end_ids
                    unk_start_record = -1
                    unk_last = False

                # if ltoken != 1 and token not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']:
                #     print(token)
                if token in ['[CLS]', '[SEP]', '[PAD]']:
                    continue
                if token[:2] == '##':
                    ltoken = ltoken - 2
                token_start, token_end = reach, reach + ltoken
                reach = token_end

                # 一段冗长的逻辑
                if start >= token_end:  # 此时一定没有碰到span位置，reached_right一定为False
                    assert not reached_right
                    continue
                elif end <= token_start:  # 此时一定以及过了span位置，reached_right一定为True
                    reached_right = True
                    assert reached_right
                    break
                else:
                    if reached_right:
                        taggers[j] = 'I-' + tag
                        # print(tokenized_seq[j])
                        # taggers[j] = 'I'
                    else:
                        taggers[j] = 'B-' + tag
                        # print(tokenized_seq[j])
                        # taggers[j] = 'B'
                        reached_right = True
        trigger_taggers.append(taggers)
    return trigger_taggers


def find_matches(content: str, tokenized_seq: [str, ]) -> {}:
    """
    为了简化处理，默认已经将空格替换为了下划线。
    因为有空格的情况实在太难处理了。太难了。太难了。
    todo 实现有空格情况
    :param content:
    :param tokenized_seq:
    :return:
    """
    assert ' ' not in content

    token2origin = {}
    origin2token = {}
    # 先给每一个位置开辟
    for i in range(len(tokenized_seq)):
        token2origin.update({i: []})
    for i in range(len(content)):
        origin2token.update({i: -1})

    reach = 0
    unk_last = False
    unk_count = 0
    for i, token in enumerate(tokenized_seq):
        ltoken = len(token)
        first_token = token[0]
        if token == '[UNK]':
            unk_count += 1
            if not unk_last:
                unk_last = True
            continue
        elif token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
        elif token[:2] == '##':
            ltoken = len(token) - 2
            first_token = token[2]
        if unk_last:
            unk_last = False
            current_reach = content.find(token[0], reach)
            token2origin[i - 1] = list(range(reach, current_reach))
            for k in range(unk_count):
                pass
            reach = current_reach
            unk_count = 0

        position = content.find(first_token, reach)
        token2origin[i] = list(range(position, position + ltoken))
        reach = position + ltoken

        assert content[position + ltoken - 1] == token[-1], str([content, position, ltoken, token, i])

    for key, value in token2origin.items():
        for v in value:
            origin2token[v] = key

    return token2origin, origin2token


if __name__ == '__main__':
    data = read_file('../data/train_base.json')
    tokenizer = BertTokenizer.from_pretrained('../' + model_path)
    tokenized_ids = list(map(lambda x: tokenizer(x['content'].lower().replace(' ', '_')).data['input_ids'], data))
    tokenized = list(map(tokenizer.convert_ids_to_tokens, tokenized_ids))
    matches = [find_matches(x['content'].lower().replace(' ', '_'), tokenized[i]) for (i, x) in enumerate(data)]
    pickle.dump(matches, open('matches.pk', 'wb'))


