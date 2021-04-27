# 模型的基本配置信息，
# 是写成分散的变量，还是构建一个config字典，还是构建多个config字典更方便?

# Path
train_file_path = 'data/train_base.json'
test_file_path = 'data/dev_base.json'
train_trans_file_path = 'data/trans_train.json'
test_trans_file_path = 'data/trans_dev.json'
model_path = '../../RoBERTa_zh_Large_PyTorch'
# model_path = 'bert-base-chinese'
# model_path = '../chinese_wwm_pytorch'
# model_path = '../chinese_wwm_ext_pytorch'
# model_path = '../chinese_wwm_ext_pytorch_ContTrain_epoch_1_batch_500_bsz_4'
inner_model = True

# Data Definitions
role_types = [
    'obj-per',  # 0
    'amount',   # 1
    'title',    # 2
    'sub-org',  # 3
    'number',   # 4
    'way',      # 5
    'collateral',   # 6
    'obj',      # 7
    'target-company',   # 8
    'share-org',    # 9
    'sub-per',  # 10
    'sub',      # 11
    'data',     # 12
    'obj-org',  # 13
    'proportion',   # 14
    'date',     # 15
    'share-per',    # 16
    'institution',  # 17
    'money'     # 18
]
role_index = {v: i for i, v in enumerate(role_types)}
event_types_init = [
    '质押',
    '股份股权转让',
    '起诉',
    '投资',
    '减持'
]
event_types_init_index = {v: i for i, v in enumerate(event_types_init)}
event_types_full = [
    '质押',
    '股份股权转让',
    '起诉',
    '投资',
    '减持',
    '收购',
    '担保',
    '中标',
    '签署合同',
    '判决'
]
event_types_full_index = {v: i for i, v in enumerate(event_types_full)}
event_available_roles = {
    '质押': {'sub-org', 'sub-per', 'obj-org', 'obj-per', 'collateral', 'date', 'money', 'number', 'proportion'},
    '股份股权转让': {'sub-org', 'sub-per', 'obj-org', 'obj-per', 'collateral', 'date', 'money', 'number', 'proportion',
               'target-company'},
    '起诉': {'sub-org', 'sub-per', 'obj-org', 'obj-per', 'date'},
    '投资': {'sub', 'obj', 'money', 'date'},
    '减持': {'sub', 'obj', 'title', 'date', 'share-per', 'share-org'},
    '收购': {'sub-org', 'sub-per', 'obj-org', 'way', 'date', 'money', 'number', 'proportion', 'target-company'},
    '担保': {'sub-org', 'sub-per', 'obj-org', 'way', 'amount', 'date'},
    '中标': {'sub', 'obj', 'amount', 'date'},
    '签署合同': {'sub-org', 'sub-per', 'obj-org', 'obj-per', 'amount', 'date'},
    '判决': {'institution', 'sub-org', 'sub-per', 'obj-org', 'obj-per', 'date', 'money'}
}

# Regex
percentage_regex = r'\d+(\.\d+)?%'
digits_regex = r'(不)?(低于|超过|超)?\d+(\.\d+)?(万|亿)?(股)?'
date_regex = r'((\d+|今|去|前)\s?年底?)?(\d+月份?)?(\d+日)?'

# Model
pos_tags = ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'k', 'm', 'n', 'nd', 'nh', 'ni', 'nl', 'ns', 'nt', 'nz', 'o',
            'p', 'q', 'r', 'u', 'v', 'wp', 'ws', 'x', 'z']
pos_tags_index = {i: x for (i, x) in enumerate(pos_tags)}

old_ner_tags = [
    'B-Nh',
    'B-Ni',
    'B-Ns',
    'E-Nh',
    'E-Ni',
    'E-Ns',
    'S-Nh',
    'S-Ni',
    'S-Ns',
    'I-Nh',
    'I-Ni',
    'I-Ns',
    'O',
]
ner_tags = ['Nh', 'Ni', 'Ns']
ner_tags_index = {i: x for (i, x) in enumerate(ner_tags)}

srl_heads = ['HEAD']
srl_tags = [
    'A0', 'A0-ADV', 'A0-CRD', 'A0-PRD', 'A0-PSE', 'A0-PSR', 'A1', 'A1-CRD', 'A1-FRQ', 'A1-PRD', 'A1-PSE', 'A1-PSR',
    'A1-QTY', 'A2', 'A2-CRD', 'A2-PSE', 'A2-PSR', 'A2-QTY', 'A3', 'A4', 'ARGM-ADV', 'ARGM-BNF', 'ARGM-CND', 'ARGM-CRD',
    'ARGM-DGR', 'ARGM-DIR', 'ARGM-DIS', 'ARGM-EXT', 'ARGM-FRQ', 'ARGM-LOC', 'ARGM-MNR', 'ARGM-PRD', 'ARGM-PRP',
    'ARGM-TMP', 'ARGM-TPC', 'rel-DIS', 'rel-EXT']
srl_tags_index = {i: x for (i, x) in enumerate(srl_tags)}

dep_tags = [
    'ADV',
     'ATT',
     'CMP',
     'COO',
     'DBL',
     'FOB',
     'HED',
     'IOB',
     'LAD',
     'POB',
     'RAD',
     'SBV',
     'VOB',
     'WP']
dep_tag_index = {i: x for (i, x) in enumerate(dep_tags)}

sdpTree_tags = [
    'AGT',
     'CONT',
     'DATV',
     'EXP',
     'FEAT',
     'LINK',
     'LOC',
     'MANN',
     'MATL',
     'MEAS',
     'PAT',
     'REAS',
     'Root',
     'SCO',
     'STAT',
     'TIME',
     'TOOL',
     'dAGT',
     'dCONT',
     'dDATV',
     'dEXP',
     'dFEAT',
     'dLINK',
     'dLOC',
     'dMANN',
     'dMATL',
     'dMEAS',
     'dPAT',
     'dREAS',
     'dSCO',
     'dSTAT',
     'dTIME',
     'dTOOL',
     'eCOO',
     'ePREC',
     'eSUCC',
     'mDEPD',
     'mNEG',
     'mPUNC',
     'mRELA',
     'rAGT',
     'rCONT',
     'rDATV',
     'rEXP',
     'rFEAT',
     'rLINK',
     'rLOC',
     'rMANN',
     'rMATL',
     'rMEAS',
     'rPAT',
     'rREAS',
     'rSCO',
     'rSTAT',
     'rTIME',
     'rTOOL']
sdpTree_tags_index = {i: x for (i, x) in enumerate(sdpTree_tags)}

sdpGraph_tags = [
    'AGT',
     'CONT',
     'DATV',
     'EXP',
     'FEAT',
     'LINK',
     'LOC',
     'MANN',
     'MATL',
     'MEAS',
     'PAT',
     'REAS',
     'Root',
     'SCO',
     'STAT',
     'TIME',
     'TOOL',
     'dAGT',
     'dCONT',
     'dDATV',
     'dEXP',
     'dFEAT',
     'dLINK',
     'dLOC',
     'dMANN',
     'dMATL',
     'dMEAS',
     'dPAT',
     'dREAS',
     'dSCO',
     'dSTAT',
     'dTIME',
     'dTOOL',
     'eCOO',
     'ePREC',
     'eSUCC',
     'mDEPD',
     'mNEG',
     'mPUNC',
     'mRELA',
     'rAGT',
     'rCONT',
     'rDATV',
     'rEXP',
     'rFEAT',
     'rLINK',
     'rLOC',
     'rMANN',
     'rMATL',
     'rMEAS',
     'rPAT',
     'rREAS',
     'rSCO',
     'rSTAT',
     'rTIME',
     'rTOOL'
]
sdpGraph_tags_index = {i: x for (i, x) in enumerate(sdpGraph_tags)}

segment_feature = False
segment_feature_cnt = 1 # 标记开始与非开始 assert token[0]为开始 非CLS

postag_feature = True
postag_feature_cnt = len(pos_tags) + 1
postag_embedding_dim = 10


ner_feature = True
ner_feature_cnt = len(ner_tags) + 1
ner_embedding_dim = 10

srl_head_feature = True
srl_head_feature_cnt = len(srl_heads) + 1 # should be 2
srl_head_embedding_dim = 5
srl_role_feature = True
srl_role_feature_cnt = len(srl_tags) + 1
srl_role_embedding_dim = 10

dep_sub_feature = True
dep_sub_feature_cnt = len(dep_tags) + 1
dep_sub_embedding_dim = 10
dep_obj_feature = True
dep_obj_feature_cnt = dep_sub_feature_cnt
dep_obj_embedding_dim = 10

sdptree_sub_feature = True
sdptree_sub_feature_cnt = len(sdpTree_tags) + 1
sdptree_sub_embedding_dim = 10
sdptree_obj_feature = True
sdptree_obj_feature_cnt = sdptree_sub_feature_cnt
sdptree_obj_embedding_dim = 10

regex_proportion_feature = False
regex_proportion_feature_cnt = 1

ltp_feature = True
ltp_feature_cnt = (segment_feature_cnt if segment_feature else 0) + \
                  (postag_feature_cnt if postag_feature else 0) + \
                  (ner_feature_cnt if ner_feature else 0) + \
                  (srl_head_feature_cnt if srl_head_feature else 0) + \
                  (srl_role_feature_cnt if srl_role_feature else 0) + \
                  (dep_sub_feature_cnt if dep_sub_feature else 0) + \
                  (dep_obj_feature_cnt if dep_obj_feature else 0) + \
                  (sdptree_sub_feature_cnt if sdptree_sub_feature else 0) + \
                  (sdptree_obj_feature_cnt if sdptree_obj_feature else 0) + \
                  (regex_proportion_feature_cnt if regex_proportion_feature else 0)
ltp_feature_cnt_fixed = ltp_feature_cnt
ltp_embedding_dim = (postag_embedding_dim if postag_feature else 0) + \
                  (ner_embedding_dim if ner_feature else 0) + \
                  (srl_head_embedding_dim if srl_head_feature else 0) + \
                  (srl_role_embedding_dim if srl_role_feature else 0) + \
                  (dep_sub_embedding_dim if dep_sub_feature else 0) + \
                  (dep_obj_embedding_dim if dep_obj_feature else 0) + \
                  (sdptree_sub_embedding_dim if sdptree_sub_feature else 0) + \
                  (sdptree_obj_embedding_dim if sdptree_obj_feature else 0)


# self atten
n_head = 6
d_head = 1024

role_alpha = 0.35
role_gamma = 2
# Train
PLMs = ['bert', 'RoBERTa']
#   RoBERTa config
config = \
{
    "attention_probs_dropout_prob": 0.1,
    "directionality": "bidi",
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "max_position_embeddings": 512,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "pooler_fc_size": 768,
    "pooler_num_attention_heads": 12,
    "pooler_num_fc_layers": 3,
    "pooler_size_per_head": 128,
    "pooler_type": "first_token_transform",
    "type_vocab_size": 2,
    "vocab_size": 21128,

    'num_labels': len(event_types_init)
}
event_detection_bsz = 8
event_detection_model = 'bert'


sentence_representation_bsz = 4

argument_extraction_bsz = 8

ner_bsz = 4
ner_lr = 2e-5
ner_epoch = 30
ner_threshold = 0.5

# Eval
train_val_split_ratio = 0.9
event_detection_theshold = 0.5
trigger_extraction_threshold = 0.5
argument_extraction_threshold = 0.5


# continue pre train
truncation_length = 255
replace_word_select_ratio = 0.9 # 选择词频前多少的词进行mask替换
mlm_bsz = 4
batch_mult_cnt = 4  # 每多少次求一次loss，相当于将bsz增大了多少倍

# label smoothing
activate_label_smoothing = True
label_smoothing_range = 3
