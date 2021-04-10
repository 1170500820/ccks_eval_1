# 模型的基本配置信息，
# 是写成分散的变量，还是构建一个config字典，还是构建多个config字典更方便?

# Path
train_file_path = 'data/train_base.json'
test_file_path = 'data/dev_base.json'
train_trans_file_path = 'data/trans_train.json'
test_trans_file_path = 'data/trans_dev.json'
# model_path = '../RoBERTa_zh_Large_PyTorch'
# model_path = 'bert-base-chinese'
# model_path = '../chinese_wwm_pytorch'
model_path = '../chinese_wwm_ext_pytorch'
# model_path = '../chinese_wwm_ext_pytorch_ContTrain_epoch_1_batch_500_bsz_4'

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

# Model
segment_feature = True
segment_feature_cnt = 1 # 标记开始与非开始 assert token[0]为开始 非CLS

postag_feature = True
postag_feature_cnt = 29

ner_feature = True
ner_feature_cnt = 13 # B,I,E,S for Nh, Ni, Ns. And O

parse_feature = False   # todo 依存与语义角色特征还没想好如何加入.
parse_feature_cnt = 0

role_feature = False
role_feature_cnt = 0

ltp_feature = True
ltp_feature_cnt = (segment_feature_cnt if segment_feature else 0) + \
                  (postag_feature_cnt if postag_feature else 0) + \
                  (ner_feature_cnt if ner_feature else 0) + \
                  (parse_feature_cnt if parse_feature else 0) + \
                  (role_feature_cnt if role_feature else 0)
ltp_feature_cnt_fixed = ltp_feature_cnt + 2 # todo pos与ner分别加入两个占位符

pos_tags = ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'k', 'm', 'n', 'nd', 'nh', 'ni', 'nl', 'ns', 'nt', 'nz', 'o',
            'p', 'q', 'r', 'u', 'v', 'wp', 'ws', 'x', 'z']
pos_tags_index = {i: x for (i, x) in enumerate(pos_tags)}

ner_tags = [
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
ner_tags_index = {i: x for (i, x) in enumerate(ner_tags)}

n_head = 6
d_head = 1024
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

argument_extraction_bsz = 4


# Eval
train_val_split_ratio = 0.9
event_detection_theshold = 0.5
trigger_extraction_threshold = 0.5


# continue pre train
truncation_length = 255
replace_word_select_ratio = 0.9 # 选择词频前多少的词进行mask替换
mlm_bsz = 4
batch_mult_cnt = 4  # 每多少次求一次loss，相当于将bsz增大了多少倍
