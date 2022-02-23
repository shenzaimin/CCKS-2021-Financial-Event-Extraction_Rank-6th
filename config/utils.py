import torch
from typing import List, Tuple
from common import Instance
import torch.optim as optim
import torch.nn as nn
from config import Config
from termcolor import colored
from transformers import BertTokenizer
from tqdm import tqdm

bert_model_dir = "chinese_roberta_wwm_ext_pytorch"  # change this if needed
tokenizer = BertTokenizer.from_pretrained(bert_model_dir, do_lower_case=True)
# bert_model_dir_2 = "chinese_roberta_wwm_large_ext_pytorch"  # change this if needed
# tokenizer_2 = BertTokenizer.from_pretrained(bert_model_dir_2, do_lower_case=True)
# print(1)

def log_sum_exp_pytorch(vec: torch.Tensor) -> torch.Tensor:
    """
    Calculate the log_sum_exp trick for the tensor.
    :param vec: [batchSize * from_label * to_label].
    :return: [batchSize * to_label]
    """
    maxScores, idx = torch.max(vec, 1)
    maxScores[maxScores == -float("Inf")] = 0
    maxScoresExpanded = maxScores.view(vec.shape[0] ,1 , vec.shape[2]).expand(vec.shape[0], vec.shape[1], vec.shape[2])
    return maxScores + torch.log(torch.sum(torch.exp(vec - maxScoresExpanded), 1))


def batching_list_instances(config: Config, insts: List[Instance]):
    """
    构造模型需要的输入，也即将文本和标签Tensor化存储
    """
    train_num = len(insts)
    batch_size = config.batch_size
    total_batch = train_num // batch_size + 1 if train_num % batch_size != 0 else train_num // batch_size
    batched_data = []
    for batch_id in tqdm(range(total_batch)):
        one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
        batched_data.append(simple_batching(config, one_batch_insts))

    return batched_data


def simple_batching(config, insts: List[Instance]) -> Tuple:

    """
    batching these instances together and return tensors. The seq_tensors for word and char contain their word id and char id.
    :return
        word_seq_tensor: Shape: (batch_size, max_seq_length)
        word_seq_len: Shape: (batch_size), the length of each sentence in a batch.
        context_emb_tensor: Shape: (batch_size, max_seq_length, context_emb_size)
        char_seq_tensor: Shape: (batch_size, max_seq_len, max_char_seq_len)
        char_seq_len: Shape: (batch_size, max_seq_len),
        label_seq_tensor: Shape: (batch_size, max_seq_length)
        annotation_mask: Shape (batch_size, max_seq_length, label_size)
    """
    batch_size = len(insts)
    batch_data = insts
    label_size = config.label_size

    # transformer, convert tokens to ids
    for idx in range(batch_size):
        batch_data[idx].word_ids = tokenizer.convert_tokens_to_ids(batch_data[idx].input.words)

    # probably no need to sort because we will sort them in the model instead.
    word_seq_len = torch.LongTensor(list(map(lambda inst: len(inst.input.words), batch_data)))
    max_seq_len = word_seq_len.max()
    max_allowed_len = min(max_seq_len, config.max_len)

    word_seq_tensor = torch.zeros((batch_size, max_allowed_len), dtype=torch.long)
    label_seq_tensor = torch.zeros((batch_size, max_allowed_len), dtype=torch.long)

    annotation_mask = None
    if batch_data[0].is_prediction is not None:
        annotation_mask = torch.zeros((batch_size, max_allowed_len, label_size), dtype=torch.long)

    for idx in range(batch_size):
        cur_len = word_seq_len[idx]  # careful here, when curr_len<=max_allowed_len, that's ok
        if cur_len <= max_allowed_len:
            word_seq_tensor[idx, :word_seq_len[idx]] = torch.LongTensor(batch_data[idx].word_ids)
            if batch_data[idx].output_ids:
                label_seq_tensor[idx, :word_seq_len[idx]] = torch.LongTensor(batch_data[idx].output_ids)

            if batch_data[idx].is_prediction is not None:
                for pos in range(len(batch_data[idx].input)):
                    if batch_data[idx].is_prediction[pos]:  # if label is true (is about to be predicted)
                        annotation_mask[idx, pos, :] = 1
                        annotation_mask[idx, pos, config.start_label_id] = 0
                        annotation_mask[idx, pos, config.stop_label_id] = 0
                    else:
                        annotation_mask[idx, pos, batch_data[idx].output_ids[pos]] = 1 # settled
                annotation_mask[idx, word_seq_len[idx]:, :] = 1
        else:
            word_seq_len[idx] = max_allowed_len
            # error
            word_seq_tensor[idx] = torch.LongTensor(batch_data[idx].word_ids[:max_allowed_len])
            if batch_data[idx].output_ids:
                label_seq_tensor[idx] = torch.LongTensor(batch_data[idx].output_ids[:max_allowed_len])

            if batch_data[idx].is_prediction is not None:
                for pos in range(len(batch_data[idx].input))[:max_allowed_len]:
                    if batch_data[idx].is_prediction[pos]:  # if label is true(is about to be predicted)
                        annotation_mask[idx, pos, :] = 1
                        annotation_mask[idx, pos, config.start_label_id] = 0
                        annotation_mask[idx, pos, config.stop_label_id] = 0
                    else:
                        annotation_mask[idx, pos, batch_data[idx].output_ids[pos]] = 1  # settled
                annotation_mask[idx, word_seq_len[idx]:, :] = 1

    word_seq_tensor = word_seq_tensor.to(config.device)
    label_seq_tensor = label_seq_tensor.to(config.device)
    word_seq_len = word_seq_len.to(config.device)
    annotation_mask = annotation_mask.to(config.device) if annotation_mask is not None else None

    return word_seq_tensor, word_seq_len, annotation_mask, label_seq_tensor


def lr_decay(config, optimizer: optim.Optimizer, epoch: int) -> optim.Optimizer:
    """
    Method to decay the learning rate
    :param config: configuration
    :param optimizer: optimizer
    :param epoch: epoch number
    :return:
    """
    lr = config.learning_rate / (1 + config.lr_decay * (epoch - 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('learning rate is set to: ', lr)
    return optimizer


def get_optimizer(config: Config, model: nn.Module):
    params = model.parameters()
    if config.optimizer.lower() == "sgd":
        print(
            colored("Using SGD: lr is: {}, L2 regularization is: {}".format(config.learning_rate, config.l2), 'yellow'))
        return optim.SGD(params, lr=config.learning_rate, weight_decay=float(config.l2))
    elif config.optimizer.lower() == "adam":
        print(colored("Using Adam", 'yellow'))
        return optim.Adam(params)
    else:
        print("Illegal optimizer: {}".format(config.optimizer))
        exit(1)

