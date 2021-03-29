# The model shared by Trigger and Argument extraction model
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from .conditional_layer_normalization import *
from settings import *


class SentenceRepresentation(nn.Module):
    def __init__(self, PLM_path, hidden_size, pass_cln=False):
        super(SentenceRepresentation, self).__init__()
        self.concatenation = SentenceTypeConcatenation(PLM_path)
        self.PLM = BertModel.from_pretrained(PLM_path)
        self.CLN = ConditionalLayerNormalization(hidden_size, pass_layer_norm=pass_cln)

    def forward(self, sentence, sentence_type):
        """
        todo 应该能够处理batch的情况了，非batch一律包装成batch再处理
        :param sentence: str or [str, ]
        :param sentence_type: str or [str, ]
        :return: [H_styp1, H_styp2,...]
        """
        if type(sentence) == str:
            sentence = [sentence]
            sentence_type = [sentence_type]
        # contextualized representation
        output = self.concatenation(sentence, sentence_type)
        input_ids, token_type_ids, attention_mask = output['input_ids'], output['token_type_ids'], \
                                                    output['attention_mask']
        output = self.PLM(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        #   output is of size (bsz, seq_l, hidden)
        # take H_s and H_c
        #   the input_ids must be <CLS> [event type] <SEP> [sentence tokens] <SEP>
        sep_positions = (input_ids == 102).nonzero().T[1].tolist()
        bsz = len(sentence)
        H_cs, H_ss = [], []
        seq_ls = []
        for s in range(bsz):
            cur_H_c = output[0][s][1:sep_positions[s * 2]] # (type_l, hidden)
            pooled_cur_H_c = torch.div(torch.sum(cur_H_c, dim=0), cur_H_c.size()[0]).unsqueeze(dim=0)    # (1, hidden)
            H_cs.append(pooled_cur_H_c)
            H_ss.append(output[0][s][sep_positions[s * 2] + 1: sep_positions[s * 2 + 1]]) # (seq_l, hidden)
            seq_ls.append(H_ss[-1].size(0))
        max_seq_l = max(seq_ls)
        for i in range(len(H_ss)):
            H_ss[i] = torch.cat([H_ss[i], torch.zeros(max_seq_l - H_ss[i].size()[0], H_ss[i].size()[1])])  # todo
        #   H_c and H_s be of size (type_l/seq_l, hidden)
        #   seq_l各有不同，所以需要补0 todo 会影响效果吗
        # MeanPooling on H_c
        # H_styp = CLN(H_s, MeanPooled H_c)
        H_styp = self.CLN(torch.stack(H_ss), torch.stack(H_cs))
        return H_styp


class SentenceTypeConcatenation(nn.Module):
    """
    简单的拼接操作
    tokenizer是可以处理batch的
    """
    def __init__(self, tokenizer_path):
        super(SentenceTypeConcatenation, self).__init__()
        self.path = tokenizer_path
        self.tokenizer = BertTokenizer.from_pretrained(self.path)

    def forward(self, sentence, sentence_type):
        """

        :param sentence: [str, ]
        :param sentence_type: [str, ]
        :return:
        """
        return self.tokenizer(sentence_type, sentence, padding=True, truncation=True, return_tensors='pt')

