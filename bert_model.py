"""
transformer based models
"""
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel

from model.linear_crf_inferencer import LinearCRF
from typing import Tuple


class BertCRF(BertPreTrainedModel):
    def __init__(self, cfig):
        super(BertCRF, self).__init__(cfig)

        # self.device = cfig.device
        self.num_labels = len(cfig.label2idx)
        self.bert = BertModel(cfig)
        self.dropout = nn.Dropout(cfig.hidden_dropout_prob)
        self.classifier = nn.Linear(cfig.hidden_size, len(cfig.label2idx))
        self.inferencer = LinearCRF(cfig)

        self.init_weights()

    def forward(self, input_ids, input_seq_lens=None, annotation_mask=None, labels=None,
                attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, add_crf=False):

        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)   # (batch_size, seq_length, hidden_size)
        logits = self.classifier(sequence_output)  # (batch_size, seq_length, num_labels)

        if labels is not None:
            batch_size = input_ids.size(0)
            sent_len = input_ids.size(1)  # one batch max seq length
            maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long).view(1, sent_len).expand(batch_size, sent_len).to(self.device)
            mask = torch.le(maskTemp, input_seq_lens.view(batch_size, 1).expand(batch_size, sent_len)).to(self.device)

            unlabed_score, labeled_score = self.inferencer(logits, input_seq_lens, labels, attention_mask)
            return unlabed_score - labeled_score

        else:
            bestScores, decodeIdx = self.inferencer.decode(logits, input_seq_lens, annotation_mask)

            return bestScores, decodeIdx

    # obsolete
    def decode(self, input_ids, input_seq_lens=None, annotation_mask=None, attention_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the batch input
        :param batchInput:
        :return:
        """
        features = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=None, position_ids=None, head_mask=None)
        features = self.dropout(features)   # (batch_size, seq_length, hidden_size)
        logits = self.classifier(features)  # (batch_size, seq_length, num_labels)

        bestScores, decodeIdx = self.inferencer.decode(logits, input_seq_lens, annotation_mask)
        return bestScores, decodeIdx


class BertCRF_pre(BertPreTrainedModel):
    def __init__(self, cfig):
        super(BertCRF_pre, self).__init__(cfig)

        # self.device = cfig.device
        self.num_labels = len(cfig.label2idx)
        self.bert = BertModel(cfig)
        self.dropout = nn.Dropout(cfig.hidden_dropout_prob)
        self.classifier_pre = nn.Linear(cfig.hidden_size, len(cfig.label2idx))
        self.inferencer_pre = LinearCRF(cfig)

        self.init_weights()

    def forward(self, input_ids, input_seq_lens=None, annotation_mask=None, labels=None,
                attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, add_crf=False):

        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)   # (batch_size, seq_length, hidden_size)
        logits = self.classifier_pre(sequence_output)  # (batch_size, seq_length, num_labels)

        if labels is not None:
            batch_size = input_ids.size(0)
            sent_len = input_ids.size(1)  # one batch max seq length
            maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long).view(1, sent_len).expand(batch_size, sent_len).to(self.device)
            mask = torch.le(maskTemp, input_seq_lens.view(batch_size, 1).expand(batch_size, sent_len)).to(self.device)

            unlabed_score, labeled_score = self.inferencer_pre(logits, input_seq_lens, labels, attention_mask)
            return unlabed_score - labeled_score

        else:
            bestScores, decodeIdx = self.inferencer_pre.decode(logits, input_seq_lens, annotation_mask)

            return bestScores, decodeIdx

    # obsolete
    def decode(self, input_ids, input_seq_lens=None, annotation_mask=None, attention_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the batch input
        :param batchInput:
        :return:
        """
        features = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=None, position_ids=None, head_mask=None)
        features = self.dropout(features)   # (batch_size, seq_length, hidden_size)
        logits = self.classifier_pre(features)  # (batch_size, seq_length, num_labels)

        bestScores, decodeIdx = self.inferencer_pre.decode(logits, input_seq_lens, annotation_mask)
        return bestScores, decodeIdx
