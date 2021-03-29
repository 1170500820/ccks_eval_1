"""
generate ltp analyze result and convert to tensor
*注意token与字之间的match转换。ltp与BertTokenizer有不同的切分策略

分词特征，只标记词开始与非词开始两种  (seq_l, 2)
词性特征，对词性包含的所有token标记    (seq_l, 28) todo 一个token一定只属于一种词性吗
ner特征，类似词性特征标记法 (seq_l, 10)
"""
from pyltp import Segmentor, Postagger, NamedEntityRecognizer, Parser, SementicRoleLabeller
import pickle
import json
from tqdm import tqdm
import torch
from settings import *


def read_file_in_ltp(filepath):
    """
    简单读取文件并转化为list
    :param filepath:
    :return:
    """
    return list(map(json.loads, open(filepath, 'r', encoding='utf-8').read().strip().split('\n')))


def print_role(roles: [], words: []):
    for role in roles:
        print(words[role.index], end=' ')
        print(role.index,
              "".join(["%s[%s]:(%d,%d)" % (arg.name, "".join(words[arg.range.start:arg.range.end + 1]), arg.range.start, arg.range.end) for arg in role.arguments]))


def generate_ltp_results():
    """
    读取train_base.json
    分词，词性标注，NER，（依存，语义角色按需加入）
    然后返回data（源数据），以及处理好的结果的list
    :return:
    """
    modelpath = '../../../ltp_data/data/'

    data = read_file_in_ltp('../data/train_base.json')
    sentences = list(map(lambda x: x['content'], data))


    # 分词
    segmentor = Segmentor()
    segmentor.load(modelpath + 'cws.model')
    segmented = [list(segmentor.segment(x.lower().replace(' ', '_'))) for x in sentences]
    segmentor.release()

    # 词性标注
    postagger = Postagger()
    postagger.load(modelpath + 'pos.model')
    posed = [list(postagger.postag(x)) for x in segmented]
    postagger.release()

    # 命名实体识别
    recognizer = NamedEntityRecognizer()
    recognizer.load(modelpath + 'ner.model')
    nered = [list(recognizer.recognize(x, posed[i])) for (i, x) in enumerate(segmented)]
    recognizer.release()

    # 依存句法分析 todo 依存句法的分析结果是一棵树，无法直接拼接到embedding上，有办法吗。
    # todo 依然要做，因为依存句法分析是语义角色标注的前置
    # parser = Parser()
    # parser.load(modelpath + 'parser.model')
    # arcs = [list(parser.parse(x, posed[i])) for (i, x) in enumerate(segmented)]
    # parser.release()


    # 语义角色标注
    # srl_labeller = SementicRoleLabeller()
    # srl_labeller.load(modelpath + 'pisrl_win.model')
    #
    # roles = [list(srl_labeller.label(x, posed[i], arcs[i])) for (i, x) in enumerate(segmented)]
    # srl_labeller.release()
    #
    # print('1\n')
    # print_role(roles[0], segmented[0])
    # print('\n2\n')
    # print_role(roles[1], segmented[1])

    # pickle.dump([segmented, posed, nered, roles], open('segmented_posed_nered_roles_0-500.pk', 'wb'))
    return data, segmented, posed, nered


def sentence_segment_match(o_data, segments):
    """
    生成原句与ltp分词结果之间的下标对应关系
    :param o_data: json来的源数据
    :param segments: 分词结果
    :return:
    """
    sentence_segment_matches = []
    for i, data in enumerate(o_data):
        sentence = data['content']
        segment = segments[i]
        segment2sentence = {}
        reach = 0
        for position, word in enumerate(segment):
            word_l = len(word)
            segment2sentence[position] = []
            for k in range(word_l):
                segment2sentence[position].append(reach + k)
            reach += word_l
        sentence2segment = {}
        for (key, value) in segment2sentence.items():
            for v in value:
                sentence2segment[v] = key
        sentence_segment_matches.append([segment2sentence, sentence2segment])
    return sentence_segment_matches


def generate_segment_feature(o_data, words, matches):
    """
    tokenized: ['[CLS]', '兴', '发', '集', '团', '发', '布', '公', '告', ',', '控', '股', '股', '东', '宜', '昌', '兴', ...]
    原句:'兴发集团发布公告,控股股东宜昌兴发集团有限责任公司于2019年11月20日将2000万股进行质押,质押方为上海浦东发展银行'
    segmented: ['兴发', '集团', '发布', '公告', ',', '控股', '股东', '宜昌', '兴发', '集团', '有限', '责任', '公司', ...]

    一个词的开始，标记为1，否则为0
    比如上面的tokenized：
        ['[CLS]', '兴', '发', '集', '团', '发', '布', '公', '告', ',', '控', '股', '股', '东', '宜', '昌', '兴', ...]
        [0, 1, 0, 1, 0, 1, 0, 1, 0, ...]
        [cls 兴发， 集团，发布，公告,...]
    :param o_data: 从train_base.json读取的源数据
    :param words: ltp分词结果，[[w1, w2, ...], ...]
    :return:
    """

    features = []
    for i, word in enumerate(words):
        sentence = o_data[i]['content'].lower().replace(' ', '_')
        token2origin, origin2token = matches[i]
        tokenized_size = len(token2origin) - 2  # todo 因为CLS也被算进来了？SEP似乎我实现的match是不包含的？
        assert sum(map(lambda x: len(x), word)) == len(sentence)    # 检查ltp分词会不会丢字或多字
        # 如果ltp不会，那么标记过程就应该不会出问题，因为tokenized与原句之间的转化是已经做过的
        word_map = []
        for w in word:
            word_length = len(w)
            word_map.append(1)
            for word_body in range(word_length - 1):
                word_map.append(0)

        feature = []
        for k in range(tokenized_size):
            # tokenized_size = len(token2origin) - 2
            span = token2origin[k + 1]  # todo 删去了第一个CLS，所以token2origin的idx应该加一
            if len(span) == 0:
                feature.append(0)   # CLS or SEP, this line should never be run
            else:
                if 1 in word_map[span[0]: span[-1] + 1]:    # 只要包含1，一律标记为开头
                    feature.append(1)
                else:
                    feature.append(0)
        features.append(feature)
    return features


def generate_postag_feature(postags, sent_seg_matches, matches):
    """
    与generate_segment_feature类似，但是生成的是词性的标记
    :param matches:
    :param postags:
    :param sent_seg_matches:
    :return:
    """
    features = []
    for i, postag in enumerate(postags):
        segment2sentence, sentence2segment = sent_seg_matches[i]
        token2origin, origin2token = matches[i]
        sequence_feature = []
        for position in range(len(token2origin) - 2):   # todo 删去CLS与SEP
            feature = [0] * (len(pos_tags) + 1)
            if len(token2origin[position + 1]) == 0:    # 每个position都应该偏移1
                feature[-1] = 1
            else:
                cur_pos = postag[sentence2segment[token2origin[position + 1][0]]]
                if cur_pos != '%':
                    feature[pos_tags.index(cur_pos)] = 1    # 默认只取token2origin的列表中第一个所对应的词性
                else:
                    feature[-1] = 1 # 词性标注结果中会出现%这样的乱码
            sequence_feature.append(feature)
        features.append(sequence_feature)
    return features


def generate_ner_features(nertags, sent_seg_matches, matches):
    features = []
    for i, nertag in enumerate(nertags):
        segment2sentence, sentence2segment = sent_seg_matches[i]
        token2origin, origin2token = matches[i]
        sequence_feature = []
        for position in range(len(token2origin) - 2):
            feature = [0] * (len(ner_tags) + 1)
            if len(token2origin[position + 1]) == 0:
                feature[-1] = 1
            else:
                cur_ner = nertag[sentence2segment[token2origin[position + 1][0]]]
                if cur_ner != '%':
                    feature[ner_tags.index(cur_ner)] = 1    # 默认只取token2origin的列表中第一个所对应的词性
                else:
                    feature[-1] = 1 # 词性标注结果中会出现%这样的乱码
            sequence_feature.append(feature)
        features.append(sequence_feature)
    return features


if __name__ == '__main__':
    data, segmented, posed, nered = generate_ltp_results()
    matches = pickle.load(open('matches.pk', 'rb'))
    segment_features = generate_segment_feature(data, segmented, matches)
    sentence_segment_matches = sentence_segment_match(data, segmented)
    postag_features = generate_postag_feature(posed, sentence_segment_matches, matches)
    ner_features = generate_ner_features(nered, sentence_segment_matches, matches)

    # tensorize
    segment_feature_tensors, postag_feature_tensors, ner_feature_tensors = [], [], []
    for s in segment_features:
        segment_feature_tensors.append(torch.tensor(s, dtype=torch.float).unsqueeze(dim=1))    # (seq_l, 1)
    for s in postag_features:
        postag_feature_tensors.append(torch.tensor(s, dtype=torch.float))  # (seq_l, 30)
    for s in ner_features:
        ner_feature_tensors.append(torch.tensor(s, dtype=torch.float)) # (seq_l, _)
    pickle.dump([segment_feature_tensors, postag_feature_tensors, ner_feature_tensors], open('syntactic_feature_tensors.pk', 'wb'))
    # 剩下只需要stack到embedding当中去即可
    # 如果需要truncation咋办？
