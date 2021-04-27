"""
generate ltp analyze result and convert to tensor
*注意token与字之间的match转换。ltp与BertTokenizer有不同的切分策略

分词特征，只标记词开始与非词开始两种  (seq_l, 2)
词性特征，对词性包含的所有token标记    (seq_l, 28) todo 一个token一定只属于一种词性吗
ner特征，类似词性特征标记法 (seq_l, 10)
"""
import sys
sys.path.append('..')
from pyltp import Segmentor, Postagger, NamedEntityRecognizer, Parser, SementicRoleLabeller
from ltp import LTP
import pickle
import json
from tqdm import tqdm
import torch
from settings import ner_tags, pos_tags, percentage_regex, srl_heads, srl_tags, dep_tags, sdpTree_tags
import re
from tqdm import tqdm


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
    parser = Parser()
    parser.load(modelpath + 'parser.model')
    arcs = [list(parser.parse(x, posed[i])) for (i, x) in enumerate(segmented)]
    parser.release()


    # 语义角色标注
    # srl_labeller = SementicRoleLabeller()
    # srl_labeller.load(modelpath + 'pisrl_win.model')
    #
    # roles = [list(srl_labeller.label(x, posed[i], arcs[i])) for (i, x) in enumerate(segmented[:500])]
    # srl_labeller.release()
    # pickle.dump(roles, open('roles0-500.pk', 'wb'))
    #
    # print('1\n')
    # print_role(roles[0], segmented[0])
    # print('\n2\n')
    # print_role(roles[1], segmented[1])

    # pickle.dump([segmented, posed, nered, arcs, roles], open('segmented_posed_nered_roles.pk', 'wb'))
    return data, segmented, posed, nered, arcs


def new_generate_ltp_results():
    # 加载模型
    ltp_model = '../../ltp_models/base1'
    ltp = LTP(path=ltp_model)

    # 读取原句子
    data = read_file_in_ltp('../data/train_base.json')
    sentences = list(map(lambda x: x['content'], data))

    segmented, pos, ner, srl, dep, sdp_tree, sdp_graph = [], [], [], [], [], [], []
    for sent in tqdm(sentences):
        # 分词
        segmented0, hidden = ltp.seg([sent])
        # 词性标注
        cur_pos = ltp.pos(hidden)
        # 命名实体识别
        cur_ner = ltp.ner(hidden)
        # 语义角色标注
        cur_srl = ltp.srl(hidden)
        # 依存句法分析
        cur_dep = ltp.dep(hidden)
        # 语义依存分析 (树)
        cur_sdp_tree = ltp.sdp(hidden, mode='tree')
        # 语义依存分析 (图)
        cur_sdp_graph = ltp.sdp(hidden, mode='graph')

        segmented.append(segmented0[0])
        pos.append(cur_pos[0])
        ner.append(cur_ner[0])
        srl.append(cur_srl[0])
        dep.append(cur_dep[0])
        sdp_tree.append(cur_sdp_tree[0])
        sdp_graph.append(cur_sdp_graph[0])

        # 生成句子与分词的对应
    sent_seg_matches = sentence_segment_match(data, segmented)
    pickle.dump([segmented, pos, ner, srl, dep, sdp_tree, sdp_graph, sent_seg_matches], open('new_ltp_results.pk', 'wb'))

    return segmented, pos, ner, srl, dep, sdp_tree, sdp_graph, sent_seg_matches


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


def tags2feature(tags: [str, ], sent_seg_matches, matches, tag_list):
    """
    token_l为该句子tokenize后去除CLS与SEP后的长度
    len(tag_set)为所有该类型tag的个数
    因为存在未知tag，所以未知tag统一被映射到最后一位 表示
    因此生成的矩阵(token_l, len(tag_set) + 1)
    每个列向量是对应token的独热类型表示

    整个过程是segment -> char -> tokens
    可以分两步进行

    len(tag_set) + 1 >= 2
    确保了不需要unsqueeze
    :param tags: 该句子与分词结果所对应的tag列表:len(tag) == len(segmented)
    :param sent_seg_matches: 该句子的分词与字符的对应dict
    :param matches: 该句子的字符与token的对应dict
    :param tag_list: 该类型tag的坐标列表
    :return: tensor: (token_l, len(tag_list) + 1)
    """
    segment2sentence, sentence2segment = sent_seg_matches
    token2origin, origin2token = matches
    # 先生成一个(token_l, len(tag_set) + 1)的全0矩阵，等待填充1(独热表示)
    matrix = torch.zeros(len(token2origin) - 2, len(tag_list) + 1, dtype=torch.float)

    # char的tag list
    char_tag_list = ['UNK'] * len(sentence2segment)
    for i, tag in enumerate(tags):
        for idx in segment2sentence[i]:
            char_tag_list[idx] = tag

    # 再写入matrix
    for i, tag in enumerate(char_tag_list):
        tag_idx = -1
        if tag in tag_list:
            tag_idx = tag_list.index(tag)
        matrix[origin2token[i] - 1][tag_idx] = 1
    return matrix


def ner2tag(ner_elem, segment_l):
    """
    将ltp的ner标注结果转化为可以输入tags2feature的tags列表
    :param ner_elem: [(ner tag, start, end), ]
    :param segment_l: length of segments
    :return: ner tag
    """
    tags = ['UNK'] * segment_l
    for elem in ner_elem:
        for idx in range(elem[1], elem[2] + 1):
            tags[idx] = elem[0]
    return tags


def pos2tag(pos_elem):
    """
    将ltp的pos标注结果转化为可以输入tags2feature的tags列表
    :param pos_elem: [pos1 str, pos2 str, pos3 str, ...]
    :return:
    """
    return pos_elem


def srl2tag(srl_elem, segment_l):
    """
    将srl标注结果转化为tags列表
    由于srl据有树结构，因此会返回两个长度均为segment_l的tags列表
    其中第一个列表是head，第二个是角色
    :param srl_elem:
    :param segment_l:
    :return:
    """
    assert len(srl_elem) == segment_l, 'ltp一定会处理成长度相等'
    head_tags, srl_tags = ['UNK'] * segment_l, ['UNK'] * segment_l
    for i, x in enumerate(srl_elem):
        if len(x) != 0:
            head_tags[i] = 'HEAD'
            for elem in x:
                for idx in range(elem[1], elem[2] + 1):
                    srl_tags[idx] = elem[0]
    return head_tags, srl_tags


def dep2tag(dep_elem, segment_l):
    """
    依存句法也是树结构，因此返回两个tags列表
    按照ltp提取的tuple的顺序，在左为主，在右为客
    先返回主，后返回客

    注意，sdpTree与dep的结构相同，因此也直接用这个函数处理了
    sdpGraph包含重复，目前暂时先不实现Graph转换为tags todo
    :param dep_elem:
    :param segment_l:
    :return:
    """
    assert len(dep_elem) == segment_l, 'ltp一定会处理成长度相等'
    sub_tags, obj_tags = ['UNK'] * segment_l, ['UNK'] * segment_l
    for i, elem in enumerate(dep_elem):
        sub_tags[i] = elem[2]
        if elem[1] != 0:
            obj_tags[elem[1] - 1] = elem[2]
    return sub_tags, obj_tags


def generate_regex_feature(data, regex_str, matches):
    """
    1 at matched area, 0 at unmatched
    :param data: origin data from trainbase.json
    :param regex_str: regex string to be searched
    :param matches:
    :return: list of (seq_l, 1)
    """
    features = []
    for i, d in enumerate(data):
        sentence = d['content'].lower().replace(' ', '_')
        token2origin, origin2token = matches[i]
        finds = re.finditer(regex_str, sentence)
        tokenized_size = len(token2origin) - 2
        token_map = [0] * tokenized_size
        for fs in finds:
            f_span = fs.span()
            # todo 检查一下
            for p in range(origin2token[f_span[0]] - 1, origin2token[f_span[1] - 1]):
                token_map[p] = 1
        features.append(token_map)
    return features


if __name__ == '__main__':
    # segmented, pos, ner, srl, dep, sdp_tree, sdp_graph, sent_seg_matches = new_generate_ltp_results()
    # Step 1
    # Load ltp results
    segmented, pos, ner, srl, dep, sdp_tree, sdp_graph, sent_seg_matches = pickle.load(open('new_ltp_results.pk', 'rb'))
    matches = pickle.load(open('matches.pk', 'rb'))

    # Step 2
    # Generate tag sequences
    pos_tag_seq = list(map(lambda x: pos2tag(x), pos))
    ner_tag_seq = list(map(lambda x: ner2tag(x[1], len(segmented[x[0]])), enumerate(ner)))
    srl_tag_head_n_roles = list(map(lambda x: srl2tag(x[1], len(segmented[x[0]])), enumerate(srl)))
    dep_tag_sub_n_obj = list(map(lambda x: dep2tag(x[1], len(segmented[x[0]])), enumerate(dep)))
    sdptree_tag_sub_n_obj = list(map(lambda x: dep2tag(x[1], len(segmented[x[0]])), enumerate(sdp_tree)))

    # Step 3
    # Generate one-hot tensor for each type of tag sequences
    syntactic_tensors = []
    for i, pos_seq in enumerate(pos_tag_seq):
        cur_sent_seg_match = sent_seg_matches[i]
        cur_match = matches[i]
        # find seq
        ner_seq = ner_tag_seq[i]
        srl_head_seq, srl_role_seq = srl_tag_head_n_roles[i]
        dep_sub_seq, dep_obj_seq = dep_tag_sub_n_obj[i]
        sdptree_sub_seq, sdptree_obj_seq = sdptree_tag_sub_n_obj[i]
        # generate tensors and concatenate
        pos_tensor = tags2feature(pos_seq, cur_sent_seg_match, cur_match, pos_tags) # (token_l, len(pos_tag_set) + 1)
        ner_tensor = tags2feature(ner_seq, cur_sent_seg_match, cur_match, ner_tags) # (token_l, len(ner_tag_set) + 1)
        srl_head_tensor = tags2feature(srl_head_seq, cur_sent_seg_match, cur_match, srl_heads)  # (token_l, len(srl_head_tag_set) + 1)
        srl_role_tensor = tags2feature(srl_role_seq, cur_sent_seg_match, cur_match, srl_tags)   # (token_l, len(srl_tag_set) + 1)
        dep_sub_tensor = tags2feature(dep_sub_seq, cur_sent_seg_match, cur_match, dep_tags) # (token_l, len(dep_tag_set) + 1)
        dep_obj_tensor = tags2feature(dep_obj_seq, cur_sent_seg_match, cur_match, dep_tags) # (token_l, len(dep_tag_set) + 1)
        sdptree_sub_tensor = tags2feature(sdptree_sub_seq, cur_sent_seg_match, cur_match, sdpTree_tags) # (token_l, len(sdptree_tag_set) + 1)
        sdptree_obj_tensor = tags2feature(sdptree_obj_seq, cur_sent_seg_match, cur_match, sdpTree_tags) # (token_l, len(sdptree_tag_set) + 1)
        syntactic_tensor = torch.cat([pos_tensor, ner_tensor, srl_head_tensor, srl_role_tensor, dep_sub_tensor, dep_obj_tensor, sdptree_sub_tensor, sdptree_obj_tensor], dim=1)
        syntactic_tensors.append(syntactic_tensor)
    pickle.dump(syntactic_tensors, open('syntactic_feature_tensors.pk', 'wb'))
    # 剩下只需要stack到embedding当中去即可
