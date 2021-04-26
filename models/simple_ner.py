from transformers import BertForTokenClassification, BertModel, BertConfig
import torch
import torch.nn as nn
import torch.nn.functional as F


class NerBert(nn.Module):
    def __init__(self, model_path=None):
        super(NerBert, self).__init__()
        self.hidden_size = BertConfig.from_pretrained(model_path).hidden_size
        self.plm = BertModel.from_pretrained(model_path)
        self.start_classifier = nn.Linear(self.hidden_size, 1)
        self.end_classifier = nn.Linear(self.hidden_size, 1)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.start_classifier.weight)
        torch.nn.init.xavier_uniform(self.end_classifier.weight)
        self.start_classifier.bias.data.fill_(0.01)
        self.end_classifier.bias.data.fill_(0.01)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        output = self.plm(input_ids=input_ids.cuda(), token_type_ids=token_type_ids.cuda(), attention_mask=attention_mask.cuda())
        starts, ends = self.start_classifier(output[0]), self.end_classifier(output[0])
        return {'results': [starts, ends]}


class NerBertLstm(nn.Module):
    def __init__(self, model_path=None):
        super(NerBertLstm, self).__init__()
        self.hidden_size = BertConfig.from_pretrained(model_path).hidden_size
        self.plm = BertModel.from_pretrained(model_path)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.start_classifier = nn.Linear(self.hidden_size * 2, 1)
        self.end_classifier = nn.Linear(self.hidden_size * 2, 1)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.start_classifier.weight)
        torch.nn.init.xavier_uniform(self.end_classifier.weight)
        self.start_classifier.bias.data.fill_(0.01)
        self.end_classifier.bias.data.fill_(0.01)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        output = self.plm(input_ids=input_ids.cuda(), token_type_ids=token_type_ids.cuda(), attention_mask=attention_mask.cuda())
        output, (_, __) = self.lstm(output[0])
        starts, ends = self.start_classifier(output), self.end_classifier(output)
        return {'results': [starts, ends]}


class NerBertLoss(nn.Module):
    def __init__(self):
        super(NerBertLoss, self).__init__()

    def forward(self, results=None, gt_label=None):
        """

        :param results: [starts, ends]
        :param gt_label: [start_label, end_label]
        :return:
        """
        nerloss = F.binary_cross_entropy_with_logits(results[0].squeeze(), gt_label[0].cuda())\
                  + F.binary_cross_entropy_with_logits(results[1].squeeze(), gt_label[1].cuda())
        return nerloss
