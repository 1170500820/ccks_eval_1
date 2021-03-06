import json
from settings import *
from transformers import BertTokenizer
from preprocess.data_process import randomize, train_val_split, multi_train_val_split
from preprocess.process_utils import *
import torch
import pickle
import random
import torch.nn.functional as F
from itertools import chain


def read_file(filepath=train_file_path):
    """
    简单读取文件并转化为list
    :param filepath:
    :return:
    """
    return list(map(json.loads, open(filepath, 'r', encoding='utf-8').read().strip().split('\n')))


def simple_batchify(train_sentences, train_types, gts, train_syntactic_features):
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
            #   pad [0] on segment feature, and pad [[0, 0, ..., 1], ] on postag and ner todo WHY?
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

    return train_sentences_batch, train_types_batch, train_gts_batch, train_syntactic_batch


def simple_bachify_for_arguments(train_sentences, train_types, train_trigs, gts, train_syntactic_features):
    train_sentences_batch, train_types_batch, train_triggers_batch, train_gts_batch, train_syntactic_batch = [], [], [], [], []
    temp_sent, temp_typ, temp_trigs, temp_gt_starts, temp_gt_ends\
        = [], [], [], [], []
    temp_synctactic_l, temp_syntactic_combine = [], []
    temp_syntactic = []
    for i, sentence in enumerate(train_sentences):
        temp_sent.append(sentence)
        temp_typ.append(train_types[i])
        temp_trigs.append(train_trigs[i])
        temp_gt_starts.append(gts[i][0].T)    # (seq_l, len(role_types)), seq_l are various
        temp_gt_ends.append(gts[i][1].T)
        # temp_syntactic_segment.append(train_syntactic_features[i][0])
        # temp_syntactic_postag.append(train_syntactic_features[i][1])
        # temp_syntactic_ner.append(train_syntactic_features[i][2])
        temp_syntactic.append(train_syntactic_features[i])  # [(seq_l1, syn_size), (seq_l1, syn_size), ...]
        temp_synctactic_l.append(temp_syntactic[-1].size(0))    # get seq_l
        if len(temp_sent) % argument_extraction_bsz == 0:
            train_sentences_batch.append(temp_sent)
            train_types_batch.append(temp_typ)
            train_triggers_batch.append(temp_trigs)
            # batchify tensors is different
            # pad zero to align them
            # each gt tensor is of size (len(role_types), seq_l), expect them to be (bsz, len(tole_types), seq_l)
            gt_starts, gt_ends \
                = torch.nn.utils.rnn.pad_sequence(list(map(lambda x: x.T, temp_gt_starts)), True, 0)\
                , torch.nn.utils.rnn.pad_sequence(list(map(lambda x: x.T, temp_gt_ends)), True, 0)    # (bsz, seq_l, len(role_types))
            # gt_starts, gt_ends = gt_starts.permute([0, 2, 1]), gt_ends.permute([0, 2, 1])   # (bsz, len(role_types), seq_l)
            train_gts_batch.append([gt_starts, gt_ends])  # both (bsz, len(role_types), seq_l)
            # pad syntactic features
            #   pad [0] on segment feature, and pad [[0, 0, ..., 1], ] on postag and ner todo 忘记这里为什么说要pad1了
            padded_syntactic_feature = torch.nn.utils.rnn.pad_sequence(temp_syntactic, True, 0) # (bsz, seq_l, syn_size)
            train_syntactic_batch.append(padded_syntactic_feature)
            temp_sent, temp_typ, temp_trigs, temp_gt_starts, temp_gt_starts, temp_gt_ends = [], [], [], [], [], []
            temp_syntactic = []

    return train_sentences_batch, train_types_batch, train_triggers_batch, train_gts_batch, train_syntactic_batch


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
        cur_sentence = d['content'].lower().replace(' ', '_')
        type_dict = {}  # key:event type  value:[trigger_span1, trigger_span2, ...]
        for e in events:
            cur_event_type = e['type']
            if cur_event_type not in type_dict:
                type_dict[cur_event_type] = []
            mentions = e['mentions']
            cur_trigger = None
            for mention in mentions:
                if mention['role'] == 'trigger':
                    cur_trigger = mention['span']  # [start, end]
            if cur_trigger is None:  # there must be one and only one trigger in a event
                raise Exception('no trigger exception')
            type_dict[cur_event_type].append(tuple(cur_trigger))    # 为了后面使用set去除重复，这里要改成tuple

        for key, value in type_dict.items():
            sentences.append(cur_sentence)
            syntactic.append([cur_segment_tensor, cur_postag_tensor, cur_ner_tensor])
            types.append(key)
            sent_match.append(cur_match)
            triggers.append(list(set(value)))  # 更改之后，现在triggers中包含多个trigger, (int, int) --> [(int, int), ...]

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
        # the data be like:
        # sent:     除 上 述 质 押 股 份 外 , 卓 众 达 富 持 有 的
        #           0  1 2 |3  4|5  6 7  8 9 10 11 12 13 14 15
        # trigger:  质押
        # span:     3, 5
        token_l_without_placeholder = len(token2origin) - 2 # todo 我没有在matches中统计结尾的SEP与CLS
        delete_token_l_without_placeholder.append(token_l_without_placeholder)
        start_tensor, end_tensor = torch.zeros(token_l_without_placeholder), torch.zeros(token_l_without_placeholder)
        for cur_span in cur_trigger:
            trigger_start, trigger_end = cur_span
            # 因为不包含CLS与SEP，所以计算出来的token coord需要减1, end还需要额外减一，
            token_start, token_end = origin2token[trigger_start] - 1, origin2token[trigger_end - 1] - 1
            start_tensor[token_start], end_tensor[token_end] = 1, 1
        gts.append([start_tensor, end_tensor])

    # split by event types:
    train_sentences_batch, train_types_batch, train_gts_batch, train_syntactic_batch = [], [], [], []
    for t in event_types_init:
        event_train_sentences, event_train_types, event_gts, event_train_syntactic_features = [], [], [], []
        for i, cur_t in enumerate(train_types):
            if cur_t == t:
                event_train_sentences.append(train_sentences[i])
                event_train_types.append(train_types[i])
                event_gts.append(gts[i])
                event_train_syntactic_features.append(train_syntactic_features[i])
        temp_train_sentences_batch, temp_train_types_batch, temp_train_gts_batch, temp_train_syntactic_batch \
            = simple_batchify(event_train_sentences, event_train_types, event_gts, event_train_syntactic_features)
        train_sentences_batch += temp_train_sentences_batch
        train_types_batch += temp_train_types_batch
        train_gts_batch += temp_train_gts_batch
        train_syntactic_batch += temp_train_syntactic_batch

    # randomize again
    zipped = list(zip(train_sentences_batch, train_types_batch, train_gts_batch, train_syntactic_batch))
    random.shuffle(zipped)
    train_sentences_batch, train_types_batch, train_gts_batch, train_syntactic_batch = zip(*zipped)
    # 经过split与randomize， 现在每个batch中的事件类型均相等

    #   simple batchify
    # train_sentences_batch, train_types_batch, train_gts_batch, train_syntactic_batch = simple_batchify(train_sentences, train_types, gts, train_syntactic_features)

    #   produce [[str, ], ] [[str, ], ] [[[tensor, tensor], ], ]
    pickle.dump([train_sentences_batch, train_types_batch, train_gts_batch, train_syntactic_batch],
                open('train_data_for_trigger_extraction.pk', 'wb'))
    #   prepare evaluating data
    spans = []  # [[], ...]
    val_syns = []
    for i, sentence in enumerate(val_sentences):
        token2origin, origin2token = val_sent_match[i]
        cur_trigger = val_triggers[i]
        cur_sent_spans = []
        for trig in cur_trigger:
            trigger_start, trigger_end = trig
            start, end = origin2token[trigger_start] - 1, origin2token[trigger_end] - 1
            cur_sent_spans.append((start, end))
        spans.append(cur_sent_spans)
        val_seg_feature, val_pos_feature, val_ner_feature = \
            val_syntactic_features[i][0], val_syntactic_features[i][1], val_syntactic_features[i][2]
        val_syns.append(torch.cat([val_seg_feature, val_pos_feature, val_ner_feature], dim=1).unsqueeze(dim=0))
    pickle.dump([val_sentences, val_types, spans, val_syns], open('val_data_for_trigger_extraction.pk', 'wb'))


def argument_extraction_reader():
    # Step 1
    # Read Data
    data = read_file()
    matches = pickle.load(open('preprocess/matches.pk', 'rb'))
    # segment_feature_tensors, postag_feature_tensors, ner_feature_tensors, regex_proportion_tensors = pickle.load(
    #     open('preprocess/syntactic_feature_tensors.pk', 'rb'))
    syntactic_tensors = pickle.load(open('preprocess/syntactic_feature_tensors.pk', 'rb'))

    # Step 2
    # Find Data
    ids, sentences, types, sent_match, trigger, arguments, syntactic = [], [], [], [], [], [], []
    # todo 会不会出现事件的trigger重复的情况，我先假设无
    repeat_cnt = 0
    total_event_cnt = 0
    for i, d in enumerate(data):
        cur_match = matches[i]
        token2origin, origin2token = cur_match
        events = d['events']
        cur_id = d['id']
        cur_syntactic_tensor = syntactic_tensors[i]
        cur_sentence = d['content'].lower().replace(' ', '_')
        type_dict = {}  # key:event type  value:[trigger_span1, trigger_span2, ...]
        argument_dict = {}  # key:event type value:[arg_lst1, arg_lst2] 与type_dict种的trigger_span相对应
        for e in events:
            cur_event_type = e['type']
            if cur_event_type not in type_dict:
                type_dict[cur_event_type] = []
                argument_dict[cur_event_type] = []
            mentions = e['mentions']
            cur_trigger = None
            args_except_trigger = []
            for mention in mentions:
                # 由于ae的trigger是传给TrigReprModel的，所以需要先转化为token上的
                cur_start, cur_end = mention['span']
                if mention['role'] == 'trigger':
                    cur_trigger = (origin2token[cur_start] - 1, origin2token[cur_end - 1] - 1)  # [start, end]
                else:
                    args_except_trigger.append(mention)
            if cur_trigger is None:  # there must be one and only one trigger in a event
                raise Exception('no trigger exception')
            type_dict[cur_event_type].append(tuple(cur_trigger))    # 为了后面使用set去除重复，这里要改成tuple
            argument_dict[cur_event_type].append(args_except_trigger)   # args_except_trigger:[mention1, mention2, ...]
            # mention: word, span, role
        for key, value in type_dict.items():
            # 对key类型的下的所有事件(trigger存放于value中)
            # triggers不去重了，因此value中可能包含重复的span，
            total_event_cnt += len(value)
            if len(value) != len(set(value)):
                repeat_cnt += 1
                print(i)
            # 下面这段代码是直接按原顺序组织arguments和trigger，即可能存在重复的sentence-type-trigger对
            # for idx, trig in enumerate(value):
            #     sentences.append(cur_sentence)
            #     ids.append(cur_id)
            #     # syntactic.append([cur_segment_tensor, cur_postag_tensor, cur_ner_tensor, cur_regex_pro_tensor])
            #     syntactic.append(cur_syntactic_tensor)
            #     types.append(key)
            #     sent_match.append(cur_match)
            #     trigger.append(trig)  # (start, end)
            #     # todo 如果两个事件的trigger相同，不如合并他们的roles作为同一个事件?
            #     # todo 如果要去重的话也不难，在此处加一个for循环重新构造spans和roles就行
            #     arguments.append(argument_dict[key][idx])
            # 下面这段代码是将拥有相同trigger的arguments都合并。
            all_argument_of_these_triggers = {}
            for idx, trig in enumerate(value):
                if trig not in all_argument_of_these_triggers:
                    all_argument_of_these_triggers[trig] = {
                        'sentence': cur_sentence,
                        'id': cur_id,
                        'syntactic': cur_syntactic_tensor,
                        'type': key,
                        'match': cur_match,
                        'args': set()
                    }
                for cur_trig_argument in argument_dict[key][idx]:
                    # cur_hashable_arg = {
                    #     'word': cur_trig_argument['word'],
                    #     'span': tuple(cur_trig_argument['span']),
                    #     'role': cur_trig_argument['role']
                    # }
                    cur_hashable_arg = (cur_trig_argument['word'], tuple(cur_trig_argument['span']),
                                        cur_trig_argument['role'])
                    all_argument_of_these_triggers[trig]['args'].add(cur_hashable_arg)
            for key1, value1 in all_argument_of_these_triggers.items():
                sentences.append(value1['sentence'])
                ids.append(value1['id'])
                syntactic.append(value1['syntactic'])
                types.append(value1['type'])
                sent_match.append(value1['match'])
                trigger.append(key1)
                arguments.append(list(map(lambda x: {'word': x[0], 'span': x[1], 'role': x[2]}, value1['args'])))



    print('repeat_cnt:', repeat_cnt)   # 类型相同，触发词相同的不同事件
    print('total event cnt:', total_event_cnt)

    # Step 3
    # Randomize
    # 打乱7次
    assert len(ids) == len(sentences) == len(types) == len(sent_match) == len(trigger) == len(arguments) == len(syntactic)
    randomize_times = 7
    for k in range(randomize_times):
        zipped = list(zip(ids, sentences, types, sent_match, trigger, arguments, syntactic))
        random.shuffle(zipped)
        ids, sentences, types, sent_match, trigger, arguments, syntactic = zip(*zipped)

    # Step 4
    # Prepare Data
    # this is argument extraction part
    # for AE training data, prepare [[sentences, ...], [types, ...], [trigger_spans, ...]]
    # gts be like starts:[argType1_sgt, argType2_sgt, ...] ends:[argType1_egt, argType2_egt, ...]
    #   starts and ends are of length len(role_types)
    #   Step 4.1
    #   split
    data_cnt = len(sentences)
    train_cnt = int(data_cnt * train_val_split_ratio)
    train_ids, train_sentences, train_types, train_sent_match\
        , train_triggers, train_arguments, train_syntactic_features\
        , val_ids, val_sentences, val_types, val_sent_match\
        , val_triggers, val_arguments, val_syntactic_features \
        = ids[:train_cnt], sentences[:train_cnt], types[:train_cnt], sent_match[:train_cnt]\
        , trigger[:train_cnt], arguments[:train_cnt], syntactic[:train_cnt]\
        , ids[train_cnt:], sentences[train_cnt:], types[train_cnt:], sent_match[train_cnt:]\
        , trigger[train_cnt:], arguments[train_cnt:], syntactic[train_cnt:]
    #   Step 4.2
    #   Prepare Training Data
    #       delete bad data. Strategy:合并
    # temp_ids, temp_sentences, temp_types, temp_match, temp_triggers, temp_arguments, temp_syntactic_features = [], [], [], [], [], [], []
    # id_trigger_type_set = set()
    # for i, sentences in enumerate(train_sentences):
    #     cur_id, cur_trigger, cur_type = train_ids[i], train_triggers[i], train_types[i]
    #     if (cur_id, cur_trigger[0], cur_trigger[1], cur_type) in id_trigger_type_set:
    #         continue
    #     else:
    #         id_trigger_type_set.add((cur_id, cur_trigger[0], cur_trigger[1], cur_type))
    #         temp_ids.append(cur_id)
    #         temp_sentences.append(sentences)
    #         temp_types.append(cur_type)
    #         temp_match.append(train_sent_match[i])
    #         temp_triggers.append(cur_trigger)
    #         temp_arguments.append(train_arguments[i])
    #         temp_syntactic_features.append(train_syntactic_features[i])
    # train_ids, train_sentences, train_types, train_sent_match, train_triggers, train_arguments, train_syntactic_features\
    #     = temp_ids, temp_sentences, temp_types, temp_match, temp_triggers, temp_arguments, temp_syntactic_features
    gts = []
    for i, sentence in enumerate(train_sentences):
        token2origin, origin2token = train_sent_match[i]
        cur_arguments = train_arguments[i]
        cur_trigger = train_triggers[i] # (start, end)
        # the data be like:
        # sent:     除 上 述 质 押 股 份 外 , 卓 众 达 富 持 有 的
        #           0  1 2 |3  4|5  6 7  8 9 10 11 12 13 14 15
        # trigger:  质押
        # span:     3, 5
        token_l_without_placeholder = len(token2origin) - 2
        start_tensors_lst, end_tensors_lst = [], []
        for role in role_types:
            cur_role_start_tensor = torch.zeros(token_l_without_placeholder)    # (seq_l)
            cur_role_end_tensor = torch.zeros(token_l_without_placeholder)
            # todo 剔除不合法结构
            for arg in cur_arguments:
                if arg['role'] == role:
                    start_index, end_index = arg['span']
                    token_start, token_end = origin2token[start_index] - 1, origin2token[end_index - 1] - 1
                    cur_role_start_tensor[token_start], cur_role_end_tensor[token_end] = 1, 1
            start_tensors_lst.append(cur_role_start_tensor)
            end_tensors_lst.append(cur_role_end_tensor)
        start_tensors, end_tensors \
            = torch.stack(start_tensors_lst), torch.stack(end_tensors_lst)   # both (len(role_types), seq_l)
        start_tensors, end_tensors \
            = start_tensors.permute([1, 0]), end_tensors.permute([1, 0])   # both (seq_l, len(role_types))
        gts.append([start_tensors, end_tensors])  # [(bsz, seq_l, len(role_types)), ~]

    # Step 5
    # Batchify
    train_sentences_batch, train_types_batch, train_triggers_batch, train_gts_batch, train_syntactic_batch \
        = simple_bachify_for_arguments(train_sentences, train_types, train_triggers, gts, train_syntactic_features)
    pickle.dump([train_sentences_batch, train_types_batch, train_triggers_batch, train_gts_batch, train_syntactic_batch], open('train_data_for_argument_extraction.pk', 'wb'))

    # Step 6
    # Prepare Evaluating Data
    #   for the evaluation data we need:
    #   1, A sentence to send into BERT
    #   2, Event types of current events to send into BERT with the sentence
    #   3, Trigger span for the shared model to perform MeanPooling
    #   4, Syntactic features to send into AEM
    #   5, Ground truth: argument spans
    arg_spans = []  # 将arguments按顺序组织, [[role1_spans, role2_spans, ..., role19_spans], ...]
    val_syns = []   # Syntactic Features
    for i, sentence in enumerate(val_sentences):
        token2origin, origin2token = val_sent_match[i]
        cur_arguments = val_arguments[i]
        cur_argument_spans = []
        for role in role_types:
            cur_role_spans = []
            for arg in cur_arguments:
                if arg['role'] == role:
                    start_index, end_index = arg['span']
                    # cur_role_spans.append(arg['span'])
                    cur_role_spans.append((origin2token[start_index] - 1, origin2token[end_index - 1] - 1))
            # 检查一下合法性
            if role in event_available_roles[val_types[i]]:
                cur_argument_spans.append(cur_role_spans)
            else:
                cur_argument_spans.append([])
        arg_spans.append(cur_argument_spans)
        val_syns.append(val_syntactic_features[i].unsqueeze(dim=0)) # (1, seq_l, syn_size)?

    pickle.dump([val_sentences, val_types, val_triggers, arg_spans, val_syns], open('val_data_for_argument_extraction.pk', 'wb'))


def ner_reader():
    ner_role_types = {
        'obj-per',
        'sub-org',
        'obj',
        'target-company',
        'sub-per',
        'sub',
        'obj-org'
    }
    tokenizer = BertTokenizer.from_pretrained(model_path)
    data = read_file()
    matches = pickle.load(open('preprocess/matches.pk', 'rb'))

    # Step 2
    # Find and SortOut
    tokenized, sentences, spans, token_spans = [], [], [], []
    # the data be like:
    # sent:     除 上 述 质 押 股 份 外 , 卓 众 达 富 持 有 的
    #           0  1 2 |3  4|5  6 7  8 9 10 11 12 13 14 15
    # trigger:  质押
    # span:     3, 5
    for i, d in enumerate(data):
        token2origin, origin2token = matches[i]
        sentences.append(d['content'].replace(' ', '_').lower())
        tokenized.append(tokenizer(d['content'].replace(' ', '_').lower(), truncation=True, padding=True, return_tensors='pt'))
        span_set = set()
        for e in d['events']:
            for r in e['mentions']:
                if r['role'] in ner_role_types:
                    span_set.add(tuple(r['span']))
        spans.append(list(span_set))
        token_spans.append(list(map(lambda x: (origin2token[x[0]], origin2token[x[1] - 1]), spans[-1])))
    # sentences: [sent_str1, sent_str2, ...]
    # spans: [[(start1, end1), (start2, end2), ...], ...]
    # matches: [[token2origin, origin2token], ...]
    # tokenized: [{'input_ids': tensor, 'token_type_ids: tensor, 'attention_mask': tensor}, ...]
    # len(sentences) == len(spans)
    assert len(tokenized) == len(sentences) == len(spans) == len(matches) == len(token_spans)

    # Step 3
    # randomize
    tokenized, sentences, spans, matches, token_spans = randomize(tokenized, sentences, spans, matches, token_spans)

    # Step 4
    # Prepare data
    train_tokenized, train_sentences, train_spans, train_matches, train_token_spans, \
    val_tokenized, val_sentences, val_spans, val_matches, val_token_spans = \
        multi_train_val_split(tokenized, sentences, spans, matches, token_spans, split_ratio=train_val_split_ratio)
    train_gt_start_tensors, train_gt_end_tensors = [], []
    for i, match in enumerate(train_matches):
        token2origin, origin2token = match
        token_l = len(token2origin)
        start_labels = [0] * token_l
        end_labels = [0] * token_l
        cur_token_spans = train_token_spans[i]
        for span in cur_token_spans:
            start_labels[span[0]] = 1
            end_labels[span[1]] = 1
        start_labels_tensor, end_labels_tensor = torch.tensor(start_labels, dtype=torch.float), torch.tensor(end_labels, dtype=torch.float)
        if end_labels_tensor.size(0) != train_tokenized[i]['input_ids'].size(1):
            print('出大问题了')
        train_gt_end_tensors.append(end_labels_tensor)
        train_gt_start_tensors.append(start_labels_tensor)

    # Step 5
    # Batchify
    train_tokenized_batch, train_sentences_batch, train_spans_batch, train_matches_batch, train_token_spans_batch, train_gt_start_tensors_batch, train_gt_end_tensors_batch = \
        batchify(train_tokenized, train_sentences, train_spans, train_matches, train_token_spans, train_gt_start_tensors, train_gt_end_tensors, bsz=ner_bsz,
                 lst_types=['dict_tensors', 'iterable', 'iterable', 'iterable', 'iterable', 'tensor', 'tensor'])
    # mid step
    # convert gt to dict of gt
    train_gt_tensors_batch = list(map(lambda x: {'gt_label': [x[1], train_gt_end_tensors_batch[x[0]]]}, enumerate(train_gt_start_tensors_batch)))
    # Step 6
    # Align training data with model
    pickle.dump(train_tokenized_batch, open('train_input.pk', 'wb'))
    pickle.dump(train_gt_tensors_batch, open('train_labels.pk', 'wb'))


    # Step 7
    # Align evaluating data with model
    pickle.dump(val_tokenized, open('val_input.pk', 'wb'))
    pickle.dump(val_token_spans, open('val_labels.pk', 'wb'))
    

def continue_pretrain_reader():
    train_base = read_file('data/train_base.json')
    train_dev = read_file('data/trans_dev.json')
    trans_base = read_file('data/trans_train.json')
    trans_dev = read_file('data/trans_dev.json')

    train_base_sent = list(map(lambda x: x['content'], train_base))
    train_dev_sent = list(map(lambda x: x['content'], train_dev))
    trans_base_sent = list(map(lambda x: x['content'], trans_base))
    trans_dev_sent = list(map(lambda x: x['content'], trans_dev))
    sentences = list(chain(trans_base_sent, trans_dev_sent, train_dev_sent, train_base_sent))

    # 去除长度大于256的句子
    max_len = 256
    sentences = list(filter(lambda x: len(x) <= max_len, sentences))

    f = open('continue_pretrain_le256_no-doc-divide.txt', 'w', encoding='utf-8')
    for sent in sentences:
        f.write(sent + '\n')
    f.close()


if __name__ == '__main__':
    # trigger_extraction_reader()
    # argument_extraction_reader()
    # ner_reader()
    continue_pretrain_reader()
