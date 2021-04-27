import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class EventDetection(nn.Module):
    def __init__(self, config):
        """

        :param config: 至少包含hidden_size, vocab_size, num_labels(需要输出的类别个数)
        """
        super(EventDetection, self).__init__()
        self.bert = BertModel.from_pretrained(config.pretrain_path)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.fill_(0.01)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = output[1]   # (bsz, 1, hidden_size)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output) # (bsz, 1, num_labels)
        return logits


class EventDetectionLoss(nn.Module):
    def __init__(self):
        super(EventDetectionLoss, self).__init__()

    def forward(self, result, gt):
        """

        :param result: (bsz, 1, num_labels)
        :param gt: (bsz, num_labels):
        :return:
        """
        reshaped_result = result.squeeze()
        loss = F.binary_cross_entropy_with_logits(result, gt)
        return loss
