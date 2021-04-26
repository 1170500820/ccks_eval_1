import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from .self_attentions import SelfAttn
from settings import role_types
from models.sentence_representation_layer import SentenceRepresentation, TriggeredSentenceRepresentation
from models.role_mask import RoleMask


class ArgumentExtractionModel(nn.Module):
    def __init__(self, n_head, hidden_size, d_head, dropout_prob, syntactic_size, skip_attn=False, skip_syn=False, skip_RPE=False, add_lstm=True):
        """

        :param n_head:
        :param hidden_size:
        :param d_head:True
        :param dropout_prob:
        :param syntactic_size:
        :param pass_attn:
        :param pass_syn:
        """
        super(ArgumentExtractionModel, self).__init__()

        self.n_head = n_head
        self.hidden_size = hidden_size
        self.d_head = d_head
        self.dropout_prob = dropout_prob
        self.syntactic_size = syntactic_size
        self.skip_attn = skip_attn
        self.skip_syn = skip_syn
        self.skip_RPE = skip_RPE
        self.add_lstm = add_lstm

        self.self_attn = SelfAttn(self.n_head, self.d_head, self.hidden_size, self.dropout_prob)

        self.syntactic_embed = nn.Linear(self.syntactic_size, self.syntactic_size, bias=False)

        if add_lstm:
            # FCN for trigger finding
            # origin + attn(origin) + syntactic + RPE
            self.fcn_start = nn.Linear(self.hidden_size , len(role_types))
            self.fcn_end = nn.Linear(self.hidden_size , len(role_types))

            # try to add a bi-LSTM layer
            self.lstm = nn.LSTM(self.hidden_size * 2 + self.syntactic_size + 1, self.hidden_size//2,
                                batch_first=True, dropout=self.dropout_prob, bidirectional=True)
        else:
            # FCN for trigger finding
            # origin + attn(origin) + syntactic + RPE
            self.fcn_start = nn.Linear(self.hidden_size * 2 + self.syntactic_size + 1, len(role_types))
            self.fcn_end = nn.Linear(self.hidden_size * 2 + self.syntactic_size + 1, len(role_types))

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.fcn_start.weight)
        torch.nn.init.xavier_uniform(self.fcn_end.weight)
        torch.nn.init.xavier_uniform(self.syntactic_embed.weight)
        self.fcn_start.bias.data.fill_(0)
        self.fcn_end.bias.data.fill_(0)

    def forward(self, cln_embeds, syntactic_structure, relative_positional_encoding):
        """

        :param cln_embeds: (bsz, seq_l, hidden_size)
        :param syntactic_structure: (bsz, seq_l, syntactic_size)
        :param relative_positional_encoding: (bsz, seq_l, 1) todo 无效区域的距离设为inf还是0
        :return:
        """
        # self attention (multihead attention)
        attn_out = self.self_attn(cln_embeds)
        syntactic_structure = self.syntactic_embed(syntactic_structure)
        # concatenation
        if self.skip_attn:
            attn_out = torch.zeros(attn_out.size()).cuda()
        if self.skip_syn:
            syntactic_structure = torch.zeros(syntactic_structure.size()).cuda()
        if self.skip_RPE:
            relative_positional_encoding = torch.zeros(relative_positional_encoding.size()).cuda()
        final_repr = torch.cat((cln_embeds, attn_out, syntactic_structure, relative_positional_encoding), dim=-1)

        if self.add_lstm:
            lstm_repr, (_, __) = self.lstm(final_repr)
            final_repr = lstm_repr

        start_logits, end_logits = self.fcn_start(final_repr), self.fcn_end(final_repr) # (bsz, seq_l, len(role_types))
        starts, ends = F.sigmoid(start_logits), F.sigmoid(end_logits)   # (bsz, seq_l, len(role_types))
        return starts, ends


class FullAEM(nn.Module):
    def __init__(self, fullAemConfig):
        """
        fullAemConfig should contains:
            - PLM_Path for PLM in SentenceRepresentation Model
            - SentenceRepresentation hidden size
            - TriggeredSentenceRepresentation hidden size
            - Argument Extraction Head:
                -- n_head
                -- d_head
                -- hidden size
                -- dropout prob
                -- syntactic size
        assume hidden size to be the same for convenient
        :param fullAemConfig:
        """
        super(FullAEM, self).__init__()
        self.repr_model = SentenceRepresentation(
            fullAemConfig.PLM_path,
            fullAemConfig.hidden_size)
        self.trigger_repr_model = TriggeredSentenceRepresentation(fullAemConfig.hidden_size)
        self.aem = ArgumentExtractionModel(
            fullAemConfig.n_head,
            fullAemConfig.hidden_size,
            fullAemConfig.d_head,
            fullAemConfig.hidden_dropout_prob,
            fullAemConfig.syntactic_size)
        self.role_mask = RoleMask(pickle.load(open(fullAemConfig.rfief_path, 'rb')))

    def forward(self, **model_input):
        """
        model_input must contains:
            - sentence: sentence batch
            - type: sentence type batch
            - trigger: trigger span batch
            - syntactic: syntactic feature batch
            - gt: ground truth batch (needed if calculate loss)
        :param model_input:
        :return:
        """
        sentence_batch, type_batch, trigger_batch, syntactic_batch, gt_batch \
            = model_input['sentence'], model_input['type'], model_input['trigger'], model_input['syntactic'], \
              model_input['gt']
        h_styp = self.repr_model(sentence_batch, type_batch)
        h_styp, RPE = self.trigger_repr_model(h_styp, trigger_batch)

