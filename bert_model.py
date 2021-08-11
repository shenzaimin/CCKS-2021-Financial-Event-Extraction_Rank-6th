"""
transformer based models
"""
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel

from model.linear_crf_inferencer import LinearCRF
from typing import Tuple
import numpy as np


class BertCRF(BertPreTrainedModel):
    def __init__(self, cfig):
        super(BertCRF, self).__init__(cfig)

        self.devices = cfig.device
        self.num_labels = len(cfig.label2idx)
        self.bert = BertModel(cfig)
        self.dynamic_bert_layer_combine = False
        self.R_Drop = False
        if self.R_Drop:
            self.R_Drop_K = cfig.R_Drop_K
        self.dropout = nn.Dropout(cfig.hidden_dropout_prob)
        # self.dropout = nn.Dropout(0.0)
        self.classifier = nn.Linear(cfig.hidden_size, len(cfig.label2idx))
        if self.dynamic_bert_layer_combine:
            self.layer_classifier = nn.Linear(cfig.hidden_size, 1)
        self.inferencer = LinearCRF(cfig)

        self.init_weights()

    def forward(self, input_ids, input_seq_lens=None, annotation_mask=None, labels=None,
                attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, add_crf=False):
        if self.dynamic_bert_layer_combine:
            outputs = self.bert(input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, output_hidden_states=True)
            # sequence_output = outputs[0]
            # 动态融合bert各层hidden_state
            all_encoder_layers = outputs[2][:-1]
            layer_logits = []
            for i, layer in enumerate(all_encoder_layers):
                print("layer: ", layer)
                layer_logits.append(self.layer_classifier(layer))
            print("np.array(layer_logits).shape:", np.array(layer_logits).shape)
            layer_logits = torch.cat((layer_logits), 2)
            print("layer_logits.shape:", layer_logits.shape)
            layer_dist = F.softmax(layer_logits, dim=2)
            print("layer_dist.shape:", layer_dist.shape)
            seq_out = torch.cat([torch.unsqueeze(x, dim=2) for x in all_encoder_layers], dim=2)
            print("seq_out.shape:", seq_out.shape)
            sequence_output = torch.matmul(torch.unsqueeze(layer_dist, dim=2), seq_out)
            sequence_output = torch.squeeze(sequence_output, dim=2)
            print("sequence_output.shape:", sequence_output.shape)
        else:
            outputs = self.bert(input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask,
                                output_hidden_states=False)
            sequence_output = outputs[0]
        sequence_output_1 = self.dropout(sequence_output)   # (batch_size, seq_length, hidden_size)
        logits = self.classifier(sequence_output_1)  # (batch_size, seq_length, num_labels)
        if labels is not None:
            batch_size = input_ids.size(0)
            sent_len = input_ids.size(1)  # one batch max seq length
            maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long).view(1, sent_len).expand(batch_size, sent_len).to(self.device)
            mask = torch.le(maskTemp, input_seq_lens.view(batch_size, 1).expand(batch_size, sent_len)).to(self.device)

            unlabed_score, labeled_score = self.inferencer(logits, input_seq_lens, labels, attention_mask)
            crf_loss = unlabed_score - labeled_score
            if self.R_Drop:
                outputs_2 = self.bert(input_ids, attention_mask=attention_mask,
                                      token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask,
                                      output_hidden_states=False)
                sequence_output_2 = outputs_2[0]
                sequence_output_2 = self.dropout(sequence_output_2)
                logits_2 = self.classifier(sequence_output_2)
                unlabed_score_2, labeled_score_2 = self.inferencer(logits_2, input_seq_lens, labels, attention_mask)
                crf_loss_2 = unlabed_score_2 - labeled_score_2
                crf_loss = 0.5 * (crf_loss + crf_loss_2)
                # if self.R_Drop:
                # sequence_output_2 = self.dropout(sequence_output)
                # logits_2 = self.classifier(sequence_output_2)
                pad_mask = torch.ones_like(attention_mask, device=self.devices).int() - attention_mask.int()
                pad_mask = torch.unsqueeze(pad_mask, dim=-1)
                pad_mask = pad_mask.gt(0)
                kl_loss = self.compute_kl_loss(sequence_output_1, sequence_output_2, pad_mask)
                # carefully choose hyper-parameters
                loss = crf_loss + self.R_Drop_K * kl_loss
                return loss
            return crf_loss

        else:
            bestScores, decodeIdx = self.inferencer.decode(logits, input_seq_lens, annotation_mask)

            return bestScores, decodeIdx

    def compute_kl_loss(self, p, q, pad_mask=None):
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss

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
