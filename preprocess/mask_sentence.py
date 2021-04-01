# functions for generating masked language
# step1 gather sentences for training, only train masked LM task, ignoring NSP todo
# step2 generate replace words. there's 0.15 * 0.1 chance that a character being replaced by another.The replace word
# cannot be too common nor too less, how to choose?
# step3 random mask and replace
#
# how to validate? l
# todo NSP该如何处理？

import json
from settings import *
from transformers import BertTokenizer
import random
import pickle
import torch


def read_file(filepath=train_file_path):
    """
    简单读取文件并转化为list
    :param filepath:
    :return:
    """
    return list(map(json.loads, open(filepath, 'r', encoding='utf-8').read().strip().split('\n')))


def gather_ccks_data():
    """
    ccks 数据的格式都是dict['content']为原句
    MLM训练也只需要原句

    :return:
    """
    train_base = read_file('../' + train_file_path)
    dev_base = read_file('../' + test_file_path)
    train_trans = read_file('../' + train_trans_file_path)
    dev_trans = read_file('../' + test_trans_file_path)

    sentences = list(map(lambda x: x['content'].lower().replace(' ', '_'), train_base + dev_base + train_trans + dev_trans))
    print(f'gather {len(sentences)} sentences')
    return sentences


def length_count(sents: [str, ]):
    """
    统计句子的长度分布
    :param sents:
    :return: {length: cnt, ...}
    """
    length_dic = {}
    for s in sents:
        l = len(s)
        if l in length_dic:
            length_dic[l] = length_dic[l] + 1
        else:
            length_dic[l] = 2
    return length_dic


def word_count(sents:[str, ]):
    """
    统计所有训练句子中的字的出现频数
    :param sents:
    :return: sorted list [(freq, word), ...]
    """
    word_dic = {}
    for s in sents:
        for w in s:
            if w in word_dic:
                word_dic[w] = word_dic[w] + 1
            else:
                word_dic[w] = 1
    lst = []
    for key, value in word_dic.items():
        lst.append((value, key))
    lst.sort(reverse=True)
    return lst


def get_replace_words(wlsts: [(int, str), ...], select_ratio=replace_word_select_ratio):
    """
    从word_count函数输出的lst中，取词频前面的词作为替换词
    :param wlsts:
    :param select_ratio:
    :return:
    """
    l = len(wlsts)
    wlsts.sort(reverse=True)
    selected_lst = wlsts[:int(l * select_ratio)]
    word_lst = list(map(lambda x: x[1], selected_lst))
    return word_lst


def perform_mask(original_ids: [int, ], replace_id: [int, ]):
    """
    对一个句子进行随机mask
    mask的各项参数均参照论文
    0.15的概率mask，其中：0.8-mask 0.1-替换为一个随机的字 0.1-不变
    需要确保mask前后，tokenize的结果长度不变

    :param original_ids: [cls, int, int, ..., sep]
    :param replace_id:
    :return:    替换后的ids [int, ...]
    """
    mask, cls, sep = 103, 101, 102

    # mask
    w_lst = [cls]
    for token in original_ids[1: -1]:
        if random.random() > 0.15:
            w_lst.append(token)
        else:
            if random.random() <= 0.8:
                w_lst.append(mask)
            elif random.random() <= 0.9:
                w_lst.append(token)
            else:
                w_lst.append(random.sample(replace_id, 1)[0])
    w_lst.append(sep)
    return w_lst


# 初始化一个tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)

# 读取句子并过滤过长的句子
sentences = gather_ccks_data()
sentences = list(filter(lambda x: len(x) <= truncation_length, sentences))

# 获得替换词
wlst = word_count(sentences)
replace_words = get_replace_words(wlst)
delete_lst = [100, 102, 101]    # 替换词不能为unk, sep, cls
replace_ids = list(filter(lambda x: x not in delete_lst, tokenizer.convert_tokens_to_ids(replace_words)))
origin_tokens, masked_tokens = [], []
for sent in sentences:
    result = tokenizer(sent)
    ids = result.data['input_ids']
    masked = perform_mask(ids, replace_ids)
    origin_tokens.append(result)
    masked_tokens.append(masked)

# randomize
zipped = list(zip(origin_tokens, masked_tokens))
random.shuffle(zipped)
origin_tokens, masked_tokens = zip(*zipped)

# batchify
input_idss, token_type_idss, attention_masks, masked_ids = [], [], [], []
temp_input_ids, temp_token_type_ids, temp_attention_masks, temp_masked = [], [], [], []
for i, b in enumerate(origin_tokens):
    m = masked_tokens[i]
    temp_input_ids.append(torch.tensor(b.data['input_ids']))
    temp_token_type_ids.append(torch.tensor(b.data['token_type_ids']))
    temp_attention_masks.append(torch.tensor(b.data['attention_mask']))
    temp_masked.append(torch.tensor(m))
    if len(temp_masked) % mlm_bsz == 0:
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(temp_input_ids, batch_first=True)
        padded_token_type_ids = torch.nn.utils.rnn.pad_sequence(temp_token_type_ids, batch_first=True)
        padded_attention_masks = torch.nn.utils.rnn.pad_sequence(temp_attention_masks, batch_first=True)
        padded_masked_ids = torch.nn.utils.rnn.pad_sequence(temp_masked, batch_first=True)
        input_idss.append(padded_input_ids)
        token_type_idss.append(padded_token_type_ids)
        attention_masks.append(padded_attention_masks)
        masked_ids.append(padded_masked_ids)
        temp_input_ids, temp_token_type_ids, temp_attention_masks, temp_masked = [], [], [], []

pickle.dump([masked_ids, temp_token_type_ids, temp_attention_masks, input_idss], open('train_data_for_mlm.pk', 'wb'))

