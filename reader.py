import json
from settings import *
from transformers import BertTokenizer
import torch
import pickle
import random
import torch.nn.functional as F


def read_file(filepath=train_file_path):
    """
    简单读取文件并转化为list
    :param filepath:
    :return:
    """
    return list(map(json.loads, open(filepath, 'r', encoding='utf-8').read().strip().split('\n')))


def event_detection_reader():
    data = read_file()
    random.shuffle(data)
    # get data for training and evaluating
    sentences, event_types = [], []
    for i, d in enumerate(data):
        sentences.append(d['content'])
        events_in_cur_sentence = []
        for event in d['events']:
            events_in_cur_sentence.append(event['type'])
        event_set = set(events_in_cur_sentence)
        event_types.append(list(event_set))

    # Train Val Split
    data_cnt = len(sentences)
    train_cnt = int(data_cnt * train_val_split_ratio)
    train_sentences, train_event_types, val_sentences, val_event_types = sentences[:train_cnt], event_types[:train_cnt] \
        , sentences[train_cnt:], event_types[train_cnt:]

    # batchify and vectorize
    tokenizer = BertTokenizer.from_pretrained(model_path)
    sent_batch, type_batch, batchified_sentence_batches, batchified_gts = [], [], [], []
    for i, sent in enumerate(train_sentences):
        sent_batch.append(sent)
        type_batch.append(train_event_types[i])
        if len(sent_batch) % event_detection_bsz == 0:
            tokenized = tokenizer(sent_batch, padding=True, truncation=True, return_tensors='pt')
            batchified_sentence_batches.append(tokenized)

            gt_batch = []
            for cur_sentence_types in type_batch:
                vec = [0] * len(event_types_init)
                for t in cur_sentence_types:
                    vec[event_types_init_index[t]] = 1
                gt_batch.append(vec)
            vectorized_gt = torch.tensor(gt_batch, dtype=torch.float)  # todo 数据类型？
            batchified_gts.append(vectorized_gt)
            sent_batch, type_batch = [], []
    pickle.dump(batchified_sentence_batches, open('sentences.pk', 'wb'))
    pickle.dump(batchified_gts, open('gts.pk', 'wb'))

    # prepare data for eval
    val_sentences_tokenized, val_gts = [], []
    for i, sent in enumerate(val_sentences):
        event_type = val_event_types[i]
        val_sentences_tokenized.append(tokenizer([sent], padding=True, truncation=True, return_tensors='pt'))
        vec = [0] * len(event_types_init)
        for t in event_type:
            vec[event_types_init_index[t]] = 1
        # val_gts.append(torch.tensor([vec], dtype=torch.float))
        val_gts.append(vec)
    pickle.dump(val_sentences_tokenized, open('val_sentences_tokenized.pk', 'wb'))
    pickle.dump(val_gts, open('val_gts.pk', 'wb'))


def trigger_extraction_reader():
    # read
    data = read_file()
    matches = pickle.load(open('preprocess/matches.pk', 'rb'))
    segment_feature_tensors, postag_feature_tensors, ner_feature_tensors = pickle.load(
        open('preprocess/syntactic_feature_tensors.pk', 'rb'))

    # find
    sentences, types, sent_match, triggers, syntactic = [], [], [], [], []
    for i, d in enumerate(data):
        cur_match = matches[i]
        events = d['events']
        cur_segment_tensor = segment_feature_tensors[i]
        cur_postag_tensor = postag_feature_tensors[i]
        cur_ner_tensor = ner_feature_tensors[i]
        for e in events:
            cur_event_type = e['type']
            mentions = e['mentions']
            cur_trigger = None
            for mention in mentions:
                if mention['role'] == 'trigger':
                    cur_trigger = [mention['span'], mention['word']]  # [span, word]
            if cur_trigger is None:  # there must be one and only one trigger in a event
                continue
            sentences.append(d['content'].lower().replace(' ', '_'))
            syntactic.append([cur_segment_tensor, cur_postag_tensor, cur_ner_tensor])
            types.append(cur_event_type)
            sent_match.append(cur_match)
            triggers.append(cur_trigger)

    # randomize
    zipped = list(zip(sentences, types, sent_match, triggers, syntactic))
    random.shuffle(zipped)
    sentences, types, sent_match, triggers, syntactic = zip(*zipped)

    # prepare data
    # for training data, prepare [[sentences, ], [types, ]]
    # and gts be like tensor(0, 0, ..., 1, 0, 0, ..., 0) tensor(0, 0, ..., 0, 0, 1, ..., 0) as start and end for a sent
    # for validation data, just use matches to convert trigger span to token type
    #   split
    data_cnt = len(sentences)
    train_cnt = int(data_cnt * train_val_split_ratio)
    train_sentences, train_types, train_sent_match, train_triggers, train_syntactic_features\
        , val_sentences, val_types, val_sent_match, val_triggers, val_syntactic_features \
        = sentences[:train_cnt], types[:train_cnt], sent_match[:train_cnt], triggers[:train_cnt], syntactic[:train_cnt]\
        , sentences[train_cnt:], types[train_cnt:], sent_match[train_cnt:], triggers[train_cnt:], syntactic[train_cnt:]
    #   prepare training data
    gts = []
    delete_token_l_without_placeholder = []
    for i, sentence in enumerate(train_sentences):
        token2origin, origin2token = train_sent_match[i]
        cur_trigger = train_triggers[i]
        trigger_start, trigger_end = cur_trigger[0]
        # the data be like:
        # sent:     除 上 述 质 押 股 份 外 , 卓 众 达 富 持 有 的
        #           0  1 2 |3  4|5  6 7  8 9 10 11 12 13 14 15
        # trigger:  质押
        # span:     3, 5
        token_l_without_placeholder = len(token2origin) - 2 # todo 我没有在matches中统计结尾的SEP与CLS
        delete_token_l_without_placeholder.append(token_l_without_placeholder)
        # 因为不包含CLS与SEP，所以计算出来的token coord需要减1, end还需要额外减一，
        token_start, token_end = origin2token[trigger_start] - 1, origin2token[trigger_end] - 2
        start_tensor, end_tensor = torch.zeros(token_l_without_placeholder), torch.zeros(token_l_without_placeholder)
        start_tensor[token_start], end_tensor[token_end] = 1, 1
        gts.append([start_tensor, end_tensor])
    #   simple batchify
    train_sentences_batch, train_types_batch, train_gts_batch, train_syntactic_batch = [], [], [], []
    temp_sent, temp_typ, temp_gt_starts, temp_gt_ends, temp_syntactic_segment, temp_syntactic_postag, temp_syntactic_ner\
        = [], [], [], [], [], [], []
    temp_synctactic_l, temp_syntactic_combine = [], []
    for i, sentence in enumerate(train_sentences):
        temp_sent.append(sentence)
        temp_typ.append(train_types[i])
        temp_gt_starts.append(gts[i][0])
        temp_gt_ends.append(gts[i][1])
        temp_syntactic_segment.append(train_syntactic_features[i][0])
        temp_syntactic_postag.append(train_syntactic_features[i][1])
        temp_syntactic_ner.append(train_syntactic_features[i][2])
        temp_synctactic_l.append(temp_syntactic_segment[-1].size()[0])
        if len(temp_sent) % sentence_representation_bsz == 0:
            train_sentences_batch.append(temp_sent)
            train_types_batch.append(temp_typ)
            # batchify tensors is different
            # pad zero to align them
            max_l = max(list(map(len, temp_gt_starts)))
            for k in range(len(temp_gt_starts)):
                temp_gt_starts[k] = F.pad(temp_gt_starts[k], [0, max_l - len(temp_gt_starts[k])])
                temp_gt_ends[k] = F.pad(temp_gt_ends[k], [0, max_l - len(temp_gt_ends[k])])
            gt_starts, gt_ends = torch.stack(temp_gt_starts), torch.stack(temp_gt_ends)
            train_gts_batch.append([gt_starts, gt_ends])  # both (bsz, seq_l)
            # pad syntactic features
            #   pad [0] on segment feature, and pad [[0, 0, ..., 1], ] on postag and ner
            max_l = max(temp_synctactic_l)
            for s_i in range(len(temp_syntactic_segment)):
                pad_l = max_l - temp_syntactic_segment[s_i].size()[0]
                if pad_l == 0:
                    continue
                pad_tensor = torch.zeros(pad_l, 1)
                temp_syntactic_segment[s_i] = torch.cat((temp_syntactic_segment[s_i], pad_tensor))  # all (max_l, 1)
            for s_i in range(len(temp_syntactic_postag)):
                pad_l = max_l - temp_syntactic_postag[s_i].size()[0]
                if pad_l == 0:
                    continue
                pad_tensor = torch.cat([torch.zeros(pad_l, postag_feature_cnt), torch.ones(pad_l, 1)], dim=1)
                temp_syntactic_postag[s_i] = torch.cat((temp_syntactic_postag[s_i], pad_tensor))    # all (max_l, pos_cnt)
            for s_i in range(len(temp_syntactic_ner)):
                pad_l = max_l - temp_syntactic_ner[s_i].size()[0]
                if pad_l == 0:
                    continue
                pad_tensor = torch.cat([torch.zeros(pad_l, ner_feature_cnt), torch.ones(pad_l, 1)], dim=1)
                temp_syntactic_ner[s_i] = torch.cat((temp_syntactic_ner[s_i], pad_tensor))  # all (max_l, ner_cnt)
            # combine syntactic features for each sentence
            for c_i in range(len(temp_syntactic_segment)):
                seg, pos, ner = temp_syntactic_segment[c_i], temp_syntactic_postag[c_i], temp_syntactic_ner[c_i]
                temp_syntactic_combine.append(torch.cat([seg, pos, ner], dim=1))
            train_syntactic_batch.append(torch.stack(temp_syntactic_combine))
            temp_sent, temp_typ, temp_gt_starts, temp_gt_ends, temp_syntactic_segment, temp_syntactic_postag, \
            temp_syntactic_ner = [], [], [], [], [], [], []
            temp_synctactic_l = []
            temp_syntactic_combine = []
    #   produce [[str, ], ] [[str, ], ] [[[tensor, tensor], ], ]
    pickle.dump([train_sentences_batch, train_types_batch, train_gts_batch, train_syntactic_batch],
                open('train_data_for_trigger_extraction.pk', 'wb'))
    #   prepare evaluating data
    spans = []
    val_syns = []
    for i, sentence in enumerate(val_sentences):
        token2origin, origin2token = val_sent_match[i]
        cur_trigger = val_triggers[i]
        trigger_start, trigger_end = cur_trigger[0]
        start, end = origin2token[trigger_start] - 1, origin2token[trigger_end] - 1
        spans.append([start, end])
        val_seg_feature, val_pos_feature, val_ner_feature = \
            val_syntactic_features[i][0], val_syntactic_features[i][1], val_syntactic_features[i][2]
        val_syns.append(torch.cat([val_seg_feature, val_pos_feature, val_ner_feature], dim=1).unsqueeze(dim=0))
    pickle.dump([val_sentences, val_types, spans, val_syns], open('val_data_for_trigger_extraction.pk', 'wb'))


if __name__ == '__main__':
    trigger_extraction_reader()
