import argparse
import random
import numpy as np
from typing import Tuple
from config import Reader, Config, evaluate_batch_insts, evaluate_batch_insts_for_entity
from config.utils import batching_list_instances
import time
import torch
from typing import List
from common import Instance
import os
import logging
import pickle
import math
import itertools
from torch.optim import Adam
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertConfig
from bert_model import BertCRF
import utils
import copy
from tqdm import tqdm
from config.reader import event_type_map, attributes_type_map
import re
from itertools import chain


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def set_seed(opt, seed):
    """
    设置seed，方便复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if opt.device.startswith("cuda"):
        print("using GPU...", torch.cuda.current_device())
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_arguments_t(parser):
    """参数配置"""
    # Training Hyperparameters
    parser.add_argument('--device', type=str, default="cuda", choices=['cpu', 'cuda'],
                        help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=2019, help="random seed")
    parser.add_argument('--digit2zero', action="store_true", default=False,
                        help="convert the number to 0, make it true is better")
    parser.add_argument('--dataset', type=str, default="data")
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=32, help="default batch size is 32 (works well)")
    parser.add_argument('--num_epochs', type=int, default=30, help="Usually we set to 10.")  # origin 100
    parser.add_argument('--train_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--dev_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--num_outer_iterations', type=int, default=10, help="Number of outer iterations for cross validation")
    parser.add_argument('--train_or_predict', type=int, default=1, help="1 means train, 2 means predict for test data")
    parser.add_argument('--train_dev_split_rate', type=float, default=0.8)
    parser.add_argument('--shuffle_reading', action="store_true", default=False,
                        help="shuffle train data before train_dev_split")
    parser.add_argument('--throw_all_O_sample', action="store_true", default=False,
                        help="丢掉全部标签为O的样本")
    parser.add_argument('--ensemble_model_num', type=int, default=1,
                        help="Number of ensemble model")



    # bert hyperparameter
    parser.add_argument('--bert_model_dir', default='bert-base-chinese-pytorch', help="Directory containing the BERT model in PyTorch")
    parser.add_argument('--max_len', default=512, help="max allowed sequence length")
    parser.add_argument('--full_finetuning', default=True, action='store_true',
                        help="Whether to fine tune the pre-trained model")
    parser.add_argument('--clip_grad', default=5, help="gradient clipping")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help="gradient accumulation")

    # model hyperparameter
    parser.add_argument('--model_folder', type=str, default="saved_model", help="The name to save the model files")
    parser.add_argument('--device_num', type=str, default='0', help="The gpu number you want to use")

    parser.add_argument('--type', type=str, default="", choices=['all'],
                        help="GPU/CPU devices")
    parser.add_argument('--dynamic_bert_layer_combine', action="store_true", default=False,
                        help="shuffle train data before train_dev_split")
    parser.add_argument('--R_Drop', action="store_true", default=False,
                        help="shuffle train data before train_dev_split")
    parser.add_argument('--R_Drop_K', type=float, default=5.00, help="R_Drop_K")

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def train_model(config: Config, train_insts: List[List[Instance]], dev_insts: List[Instance]):
    train_num = sum([len(insts) for insts in train_insts])
    logging.info(("[Training Info] number of instances: %d" % (train_num)))
    # get the batched data
    dev_batches = batching_list_instances(config, dev_insts)

    model_folder = config.model_folder

    logging.info("[Training Info] The model will be saved to: %s" % (model_folder))
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    num_outer_iterations = config.num_outer_iterations

    for iter in range(num_outer_iterations):

        logging.info(f"[Training Info] Running for {iter}th large iterations.")

        model_names = []  # model names for each fold

        train_batches = [batching_list_instances(config, insts) for insts in train_insts]

        logging.info("length of train_insts：%d"% len(train_insts))

        # train 2 models in 2 folds
        if iter != 0:
            for fold_id, folded_train_insts in enumerate(train_insts):
                logging.info(f"[Training Info] Training fold {fold_id}.")
                # Initialize bert model
                logging.info("Initialized from pre-trained Model")

                model_name = model_folder + f"/bert_crf_{fold_id}"
                model_names.append(model_name)
                train_one(config=config, train_batches=train_batches[fold_id],
                          dev_insts=dev_insts, dev_batches=dev_batches, model_name=model_name)

        # assign prediction to other folds
        logging.info("\n\n")
        logging.info("[Data Info] Assigning labels")
        torch.cuda.empty_cache()
        # using the model trained in one fold to predict the result of another fold's data
        # and update the label of another fold with the predict result
        for fold_id, folded_train_insts in enumerate(train_insts):

            cfig_path = os.path.join(config.bert_model_dir, 'bert_config.json')
            cfig = BertConfig.from_json_file(cfig_path)
            cfig.device = config.device
            cfig.label2idx = config.label2idx
            cfig.label_size = config.label_size
            cfig.idx2labels = config.idx2labels
            # dynamic_bert_layer_combine
            if opt.dynamic_bert_layer_combine:
                cfig.dynamic_bert_layer_combine = True
                print("*dynamic_bert_layer_combine*")
            else:
                cfig.dynamic_bert_layer_combine = False
            model_name = model_folder + f"/bert_crf_{fold_id}"
            model = BertCRF(cfig=cfig)
            model.to(cfig.device)
            utils.load_checkpoint(os.path.join(model_name, 'best.pth.tar'), model)

            hard_constraint_predict(config=config, model=model,
                                    fold_batches=train_batches[1 - fold_id],
                                    folded_insts=train_insts[1 - fold_id])  # set a new label id, k is set to 2, so 1 - fold_id can be used
        logging.info("\n\n")

        logging.info("[Training Info] Training the final model")

        # merge the result data to training the final model
        all_train_insts = list(itertools.chain.from_iterable(train_insts))

        logging.info("Initialized from pre-trained Model")

        model_name = model_folder + "/final_bert_crf"
        config_name = model_folder + "/config.conf"

        all_train_batches = batching_list_instances(config=config, insts=all_train_insts)
        # train the final model
        model = train_one(config=config, train_batches=all_train_batches, dev_insts=dev_insts, dev_batches=dev_batches,
                          model_name=model_name, config_name=config_name)
        # load the best final model
        utils.load_checkpoint(os.path.join(model_name, 'best.pth.tar'), model)
        model.eval()
        logging.info("\n")
        result = evaluate_model(config, model, dev_batches, "dev", dev_insts)
        logging.info("\n\n")


def hard_constraint_predict(config: Config, model: BertCRF, fold_batches: List[Tuple], folded_insts:List[Instance], model_type:str = "hard"):
    """using the model trained in one fold to predict the result of another fold"""
    batch_id = 0
    batch_size = config.batch_size
    model.eval()
    for batch in tqdm(fold_batches):
        one_batch_insts = folded_insts[batch_id * batch_size:(batch_id + 1) * batch_size]

        input_ids, input_seq_lens, annotation_mask, labels = batch
        input_masks = input_ids.gt(0)
        # get the predict result
        batch_max_scores, batch_max_ids = model(input_ids, input_seq_lens=input_seq_lens,
                                                annotation_mask=annotation_mask, labels=None, attention_mask=input_masks)

        batch_max_ids = batch_max_ids.cpu().numpy()
        word_seq_lens = batch[1].cpu().numpy()
        for idx in range(len(batch_max_ids)):
            length = word_seq_lens[idx]
            prediction = batch_max_ids[idx][:length].tolist()
            prediction = prediction[::-1]
            # update the labels of another fold
            one_batch_insts[idx].output_ids = prediction
        batch_id += 1


def train_one(config: Config, train_batches: List[Tuple], dev_insts: List[Instance], dev_batches: List[Tuple],
              model_name: str, config_name: str = None) -> BertCRF:

    # load config for bertCRF
    cfig_path = os.path.join(config.bert_model_dir,
                             'bert_config.json')
    cfig = BertConfig.from_json_file(cfig_path)
    cfig.device = config.device
    cfig.label2idx = config.label2idx
    cfig.label_size = config.label_size
    cfig.idx2labels = config.idx2labels
    # dynamic_bert_layer_combine
    if opt.dynamic_bert_layer_combine:
        cfig.dynamic_bert_layer_combine = True
        print("*dynamic_bert_layer_combine*")
    else:
        cfig.dynamic_bert_layer_combine = False
    # load pretrained bert model
    model = BertCRF.from_pretrained(config.bert_model_dir, config=cfig)
    model.to(config.device)

    if config.full_finetuning:
        logging.info('full finetuning')
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]

    else:
        logging.info('tuning downstream layer')
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]

    optimizer = Adam(optimizer_grouped_parameters, lr=config.learning_rate)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch))
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=1)

    model.train()

    epoch = config.num_epochs
    best_dev_f1 = -1
    for i in range(1, epoch + 1):
        epoch_loss = 0
        start_time = time.time()
        model.zero_grad()

        for index in tqdm(np.random.permutation(len(train_batches))):  # disorder the train batches
            model.train()
            scheduler.step()
            input_ids, input_seq_lens, annotation_mask, labels = train_batches[index]
            input_masks = input_ids.gt(0)
            # update loss
            loss = model(input_ids, input_seq_lens=input_seq_lens, annotation_mask=annotation_mask,
                         labels=labels, attention_mask=input_masks)
            epoch_loss += loss.item()
            model.zero_grad()
            loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
            optimizer.step()
        end_time = time.time()
        logging.info("Epoch %d: %.5f, Time is %.2fs" % (i, epoch_loss, end_time - start_time))

        model.eval()
        with torch.no_grad():
            # metric is [precision, recall, f_score]
            dev_metrics = evaluate_model(config, model, dev_batches, "dev", dev_insts)
            if dev_metrics[2] > best_dev_f1:  # save the best model
                logging.info("saving the best model...")
                best_dev_f1 = dev_metrics[2]

                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                optimizer_to_save = optimizer
                utils.save_checkpoint({'epoch': epoch + 1,
                                       'state_dict': model_to_save.state_dict(),
                                       'optim_dict': optimizer_to_save.state_dict()},
                                      is_best=dev_metrics[2] > 0,
                                      checkpoint=model_name)

                # Save the corresponding config as well.
                if config_name:
                    f = open(config_name, 'wb')
                    pickle.dump(config, f)
                    f.close()
        model.zero_grad()

    return model


def evaluate_model(config: Config, model: BertCRF, batch_insts_ids, name: str, insts: List[Instance]):
    # evaluation
    metrics = np.asarray([0, 0, 0], dtype=int)
    batch_id = 0
    batch_size = config.batch_size
    for batch in batch_insts_ids:
        one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]

        input_ids, input_seq_lens, annotation_mask, labels = batch
        input_masks = input_ids.gt(0)
        # get the predict result
        #print(input_ids)
        batch_max_scores, batch_max_ids = model(input_ids, input_seq_lens=input_seq_lens, annotation_mask=annotation_mask,
                         labels=None, attention_mask=input_masks)
        #print(batch_max_ids)
        metrics += evaluate_batch_insts(batch_insts=one_batch_insts,
                                        batch_pred_ids=batch_max_ids,
                                        batch_gold_ids=batch[-1],
                                        word_seq_lens=batch[1], idx2label=config.idx2labels)
        batch_id += 1
    # calculate the precision, recall and f1 score
    p, total_predict, total_entity = metrics[0], metrics[1], metrics[2]
    precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
    recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
    fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    logging.info("[%s set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (name, precision, recall, fscore))
    return [precision, recall, fscore]


def evaluate_model_for_entity(config: Config, model: BertCRF, batch_insts_ids, name: str, insts: List[Instance]):
    type_dict = {'all': [
    'afcs',
    'shr',
    'shrsf',
    'sfzh',
    'xyr',
    'afsj',
    'zsje',
    'sapt',
    'yhkh',
    'zfqd',
    'ddh',
    'sjh',
    'jyh']}

    # evaluation
    metrics = np.zeros([len(type_dict[config.type]),3],dtype=int)
    # metrics = np.asarray([0, 0, 0], dtype=int)
    batch_id = 0
    batch_size = config.batch_size
    for batch in batch_insts_ids:
        one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]

        input_ids, input_seq_lens, annotation_mask, labels = batch
        input_masks = input_ids.gt(0)
        # get the predict result
        batch_max_scores, batch_max_ids = model(input_ids, input_seq_lens=input_seq_lens, annotation_mask=annotation_mask,
                         labels=None, attention_mask=input_masks)

        metrics += evaluate_batch_insts_for_entity(batch_insts=one_batch_insts,
                                        batch_pred_ids=batch_max_ids,
                                        batch_gold_ids=batch[-1],
                                        word_seq_lens=batch[1], idx2label=config.idx2labels,
                                        type=config.type)
        batch_id += 1
    for i in range(len(type_dict[config.type])):
        p, total_predict, total_entity = metrics[i][0], metrics[i][1],metrics[i][2]
    # calculate the precision, recall and f1 score
    # p, total_predict, total_entity = metrics[0], metrics[1], metrics[2]
        precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
        recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
        fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
        logging.info("[* %s *] Precision: %.2f, Recall: %.2f, F1: %.2f" % (type_dict[config.type][i], precision, recall, fscore))
    return 0


def main():
    """
    模型训练
    """
    logging.info("Transformer implementation")
    parser = argparse.ArgumentParser(description="Transformer CRF implementation")
    opt = parse_arguments_t(parser)
    conf = Config(opt)
    conf.train_file = conf.dataset + "/train_fix"
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device_num
    # data reader
    reader = Reader(conf.digit2zero)
    set_seed(opt, conf.seed)

    if not os.path.exists(conf.model_folder):
        os.makedirs(conf.model_folder)

    # set logger
    utils.set_logger(os.path.join(conf.model_folder, 'train.log'))

    # params
    for k in opt.__dict__:
        logging.info(k + ": " + str(opt.__dict__[k]))

    # read trains/devs
    logging.info("\n")
    logging.info("Loading the datasets...")
    trains_add_devs = reader.read_txt(conf.train_file, conf.train_num, opt.type, int(conf.max_len) - 12)  # [[inst_sub1,inst_sub2],[],...]  inst:
    logging.info("Building label idx ...")
    # build label2idx and idx2label
    conf.build_label_idx(list(chain.from_iterable(trains_add_devs)))
    if opt.shuffle_reading:
        logging.info("\n")
        logging.info(f"Shuffle reading the datasets...Seed:{conf.seed}")
        random.shuffle(trains_add_devs)
    trains = list(chain.from_iterable(trains_add_devs[:int(opt.train_dev_split_rate*len(trains_add_devs))]))  # [inst,inst,...]
    devs = list(chain.from_iterable(trains_add_devs[int(opt.train_dev_split_rate*len(trains_add_devs)):]))  # [inst,inst,...]
    # if opt.throw_all_O_sample:
    #     trains = [t for t in trains if len(t.mentions) != 0]  # 去掉权威O的样本
    #     # devs = [d for d in devs if len(d.mentions) != 0]  # 去掉权威O的样本
    print('【trains: ' + str(len(trains)) + ' devs: ' + str(len(devs)) + '】')
    # import numpy as np
    # import pandas as pd
    # id_list = np.asarray([t.id for t in trains])
    # df = pd.DataFrame(id_list, index=None, columns=['质押'])
    # df.to_csv('./减持t.csv')
    # id_list_d = np.asarray([t.id for t in devs])
    # df_d = pd.DataFrame(id_list_d, index=None, columns=['质押'])
    # df_d.to_csv('./减持d.csv')

    # for type in ['质押', '股份股权转让', '起诉', '投资', '减持', '收购', '判决']:
    #     trains = trains_add_devs[:int(0.8 * len(trains_add_devs))]
    #     devs = trains_add_devs[int(0.8 * len(trains_add_devs)):]
    #     train_id_list = np.asarray([t.id for t in trains])
    #     devs_id_list = np.asarray([d.id for d in devs])




    random.shuffle(trains)
    # train model
    train_num = len(trains)
    logging.info(("[Training Info] number of instances: %d" % (train_num)))
    # get the batched data
    train_batches = batching_list_instances(conf, trains)
    dev_batches = batching_list_instances(conf, devs)

    # Initialize bert model
    logging.info("Initialized from pre-trained Model")
    model_folder = conf.model_folder
    model_name = model_folder + "/final_bert_crf"
    logging.info("[Training Info] The model will be saved to: %s" % (model_folder))
    config_name = model_folder + "/config.conf"
    # load config for bertCRF
    cfig_path = os.path.join(conf.bert_model_dir, 'bert_config.json')
    cfig = BertConfig.from_json_file(cfig_path)
    cfig.device = conf.device
    cfig.label2idx = conf.label2idx
    cfig.label_size = conf.label_size
    cfig.idx2labels = conf.idx2labels
    # dynamic_bert_layer_combine
    if opt.dynamic_bert_layer_combine:
        cfig.dynamic_bert_layer_combine = True
        logging.info("*dynamic_bert_layer_combine*")
    else:
        cfig.dynamic_bert_layer_combine = False
    # R-Drop
    if opt.R_Drop:
        cfig.R_Drop = True
        cfig.R_Drop_K = opt.R_Drop_K
        logging.info(f"*R_Drop: K={opt.R_Drop_K}*")
    else:
        cfig.R_Drop = False
    # load pretrained bert model
    # cfig.hidden_dropout_prob = 0.3
    model = BertCRF.from_pretrained(conf.bert_model_dir, config=cfig)  # 加载原始Roberta-wwm权重
    model.to(cfig.device)
    if conf.full_finetuning:
        logging.info('full finetuning')
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]

    else:
        logging.info('tuning downstream layer')
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]

    optimizer = Adam(optimizer_grouped_parameters, lr=conf.learning_rate)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.001 * epoch))

    model.train()

    epoch = conf.num_epochs
    best_dev_f1 = -1
    """开始训练"""
    for i in range(1, epoch + 1):
        epoch_loss = 0
        start_time = time.time()
        model.zero_grad()
        print('{} optim: {}'.format(i, optimizer.param_groups[0]['lr']))
        for step, index in tqdm(enumerate(np.random.permutation(len(train_batches)))):  # disorder the train batches
            model.train()
            # scheduler.step()
            input_ids, input_seq_lens, annotation_mask, labels = train_batches[index]
            input_masks = input_ids.gt(0)
            # update loss
            loss = model(input_ids, input_seq_lens=input_seq_lens, annotation_mask=annotation_mask,
                         labels=labels, attention_mask=input_masks)
            if opt.gradient_accumulation_steps > 1:  # 梯度累加
                loss = loss/opt.gradient_accumulation_steps
            epoch_loss += loss.item()
            # model.zero_grad()
            loss.backward()
            if (step + 1) % opt.gradient_accumulation_steps == 0:
                # gradient clipping
                #print(loss)
                nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=conf.clip_grad)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
        # scheduler.step()
        end_time = time.time()
        logging.info("Epoch %d: %.5f, Time is %.2fs" % (i, epoch_loss / step, end_time - start_time))

        model.eval()
        with torch.no_grad():
            # metric is [precision, recall, f_score]
            dev_metrics = evaluate_model(conf, model, dev_batches, "dev", devs)  # 模型评估
            if dev_metrics[2] > best_dev_f1:  # save the best model
                logging.info("saving the best model...")
                best_dev_f1 = dev_metrics[2]

                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                optimizer_to_save = optimizer
                utils.save_checkpoint({'epoch': epoch + 1,
                                       'state_dict': model_to_save.state_dict(),
                                       'optim_dict': optimizer_to_save.state_dict()},
                                      is_best=dev_metrics[2] > 0,
                                      checkpoint=model_name)

                # Save the corresponding config as well.
                if config_name:
                    f = open(config_name, 'wb')
                    pickle.dump(conf, f)
                    f.close()
        model.zero_grad()
    # load the best final model
    utils.load_checkpoint(os.path.join(model_name, 'best.pth.tar'), model)
    model.eval()
    logging.info("\n")
    result = evaluate_model(conf, model, dev_batches, "dev", devs)
    evaluate_model_for_entity(conf, model, dev_batches, "dev", devs)
    logging.info("\n\n")


def main_predict():
    """模型预测"""
    attributes_type_map_inv = dict([(v,k) for (k,v) in attributes_type_map.items()])
    logging.info("Transformer implementation")
    parser = argparse.ArgumentParser(description="Transformer CRF implementation")
    opt = parse_arguments_t(parser)
    conf = Config(opt)
    conf.train_file = conf.dataset + "/train_fix"
    conf.test_file = conf.dataset + "/dev"
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device_num

    # data reader
    reader = Reader(conf.digit2zero)

    # 读取分类结果
    # import pandas as pd
    # cls_out = pd.read_csv('CCKS-Cls/test_output/cls_out_test.csv')
    # cls_dict = dict()
    # for index, row in tqdm(cls_out.iterrows()):
    #     ids = row["id"]
    #     for tp in ['zy', 'gfgqzr', 'qs', 'tz', 'jc', 'sg', 'pj']:
    #         if row[tp] == 1:
    #             data_type = type_map[tp]
    #             if not ids in cls_dict.keys():
    #                 cls_dict[ids] = [data_type]
    #             else:
    #                 cls_dict[ids].append(data_type)
    # 分模型进行预测
    lst = []
    for suffix in ['x']:
        # read tests
        logging.info("\n")
        logging.info("Loading the datasets...")
        trains = reader.read_txt(conf.train_file, conf.train_num)
        trains = list(itertools.chain(*trains))
        all_tests = reader.read_test_txt(conf.test_file, conf.train_num)
        # for idx in range(len(all_tests)):
        #     # 给对象打分类标签
        #     if all_tests[idx].id in cls_dict.keys():
        #         all_tests[idx].type = cls_dict[all_tests[idx].id]
        #     else:
        #         all_tests[idx].type = []
        # tests = [test for test in all_tests if type_map[suffix] in test.type]
        tests = all_tests
        # query_list = reader.get_origin_query(conf.test_file, conf.train_num)
        # assert len(query_list) == len(tests)

        logging.info("Building label idx ...")
        # build label2idx and idx2label
        conf.build_label_idx(trains)

        # load model
        cfig_path = os.path.join(conf.bert_model_dir, 'bert_config.json')
        cfig = BertConfig.from_json_file(cfig_path)
        cfig.device = conf.device
        cfig.label2idx = conf.label2idx
        cfig.label_size = conf.label_size
        cfig.idx2labels = conf.idx2labels
        model_folder = conf.model_folder
        model_name = model_folder + "/final_bert_crf"
        # model = BertCRF.from_pretrained(conf.bert_model_dir, config=cfig)
        model = BertCRF(cfig=cfig)
        model.to(cfig.device)
        #print(os.path.join(model_name, 'best.pth.tar'))
        utils.load_checkpoint(os.path.join(model_name, 'best.pth.tar'), model)
        model.eval()
        print(os.path.join(model_name, 'best.pth.tar'))

        # predict
        test_batches = batching_list_instances(conf, tests)
        hard_constraint_predict(config=conf, model=model, fold_batches=test_batches, folded_insts=tests)
        import joblib
        joblib.dump(tests, f'{conf.model_folder}/tests.bin')
        for idx in range(len(tests)):
            prediction = tests[idx].output_ids
            prediction = [cfig.idx2labels[l] for l in prediction]
            tests[idx].prediction = prediction
        for idx in range(len(tests)):
            qids_original = tests[idx].id
            qids = re.search('\d+', qids_original).group()
            sub_id = re.search('\d+$', qids_original).group()
            start = -1
            for i in range(len(tests[idx].prediction)):
                if tests[idx].prediction[i].startswith("B-") and start == -1:
                    start = i
                if tests[idx].prediction[i].startswith("E-") and start != -1:
                    if tests[idx].prediction[i][2:] == tests[idx].prediction[start][
                                                       2:]:  # START 和 END 的类别必须保持一致，否则不能算实体，放弃抽取
                        end = i
                        value = tests[idx].seg_content[start:end+1]
                        role = attributes_type_map_inv[tests[idx].prediction[i][2:]]
                        sample = {"text_id": qids, "attributes": [
                                                              {"entity": value, "start": start+500*int(sub_id), "end": end+500*int(sub_id),
                                                               "type": role}
                                                                ]
                                  }
                        lst.append(sample)
                        start = -1
                    else:
                        start = -1
    sub_data = open(f'{conf.model_folder}/系统之神与我同在_valid_result.txt', 'w+', encoding='utf-8')
    official_test_df = open('data/dev/ccks_task1_eval_data.txt', 'r', encoding='utf-8').readlines()
    # official_test_transfer_df = open('data/dev/trans_dev.json', 'r', encoding='utf-8').readlines()
    # official_test_df.extend(official_test_transfer_df)
    merge_dict = dict()
    # 获取所有的id集合
    idx_list = []
    import json
    for line in tqdm(official_test_df):
        line = line.strip()
        line = json.loads(line)
        ids = line['text_id']
        idx_list.append(ids)
    for k in tqdm(lst):
        sam_dic = {"text_id": k['text_id'], "attributes": [k['attributes'][0]]}
        if k['text_id'] not in merge_dict.keys():
            merge_dict[k['text_id']] = sam_dic
        else:
            merge_dict[k['text_id']]["attributes"].append(k['attributes'][0])
        if k['text_id'] in idx_list:
            idx_list.remove(k['text_id'])
    merge_lst = list(merge_dict.values())
    for ids in tqdm(idx_list):
        merge_lst.append({"text_id": ids, "attributes": []})
    for i in tqdm(merge_lst):
        # ids = i['text_id']
        # events = i['attribute']
        # sub_dic = {}
        # info_dic = {}
        # for d in events:
        #     if d['type'] not in info_dic:
        #         info_dic[d['type']] = d['mentions']
        #         list1 = info_dic[d['type']]
        #     else:
        #         info_dic[d['type']] = info_dic[d['type']] + d['mentions']
        # sub_dic['text_id'] = ids
        # t_list = []
        # for key, value in info_dic.items():
        #     dic1 = {}
        #     dic1['type'] = key
        #     dic1['mentions'] = value
        #     t_list.append(dic1)
        # sub_dic['attribute'] = t_list
        # print(sub_dic)
        json.dump(i, sub_data, ensure_ascii=False)
        sub_data.write('\n')


def eval_err_output():
    attributes_type_map_inv = dict([(v, k) for (k, v) in attributes_type_map.items()])
    logging.info("Transformer implementation")
    parser = argparse.ArgumentParser(description="Transformer CRF implementation")
    opt = parse_arguments_t(parser)
    conf = Config(opt)
    conf.train_file = conf.dataset + "/train_fix"
    conf.test_file = conf.dataset + "/dev"
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device_num

    # data reader
    reader = Reader(conf.digit2zero)

    # 读取分类结果
    # import pandas as pd
    # cls_out = pd.read_csv('CCKS-Cls/test_output/cls_out_test.csv')
    # cls_dict = dict()
    # for index, row in tqdm(cls_out.iterrows()):
    #     ids = row["id"]
    #     for tp in ['zy', 'gfgqzr', 'qs', 'tz', 'jc', 'sg', 'pj']:
    #         if row[tp] == 1:
    #             data_type = type_map[tp]
    #             if not ids in cls_dict.keys():
    #                 cls_dict[ids] = [data_type]
    #             else:
    #                 cls_dict[ids].append(data_type)
    # 分模型进行预测
    lst = []
    for suffix in ['x']:
        # read trains/devs
        logging.info("\n")
        logging.info("Loading the datasets...")
        trains_add_devs = list(chain.from_iterable(reader.read_txt(conf.train_file, conf.train_num, opt.type, int(conf.max_len) - 12)))
        # all_tests = reader.read_test_txt(conf.test_file, conf.train_num)
        # for idx in range(len(all_tests)):
        #     # 给对象打分类标签
        #     if all_tests[idx].id in cls_dict.keys():
        #         all_tests[idx].type = cls_dict[all_tests[idx].id]
        #     else:
        #         all_tests[idx].type = []
        # tests = [test for test in all_tests if type_map[suffix] in test.type]
        tests = trains_add_devs
        # query_list = reader.get_origin_query(conf.test_file, conf.train_num)
        # assert len(query_list) == len(tests)

        logging.info("Building label idx ...")
        # build label2idx and idx2label
        conf.build_label_idx(trains_add_devs)

        # load model
        cfig_path = os.path.join(conf.bert_model_dir, 'bert_config.json')
        cfig = BertConfig.from_json_file(cfig_path)
        cfig.device = conf.device
        cfig.label2idx = conf.label2idx
        cfig.label_size = conf.label_size
        cfig.idx2labels = conf.idx2labels
        model_folder = conf.model_folder
        model_name = model_folder + "/final_bert_crf"
        # model = BertCRF.from_pretrained(conf.bert_model_dir, config=cfig)
        model = BertCRF(cfig=cfig)
        model.to(cfig.device)
        # print(os.path.join(model_name, 'best.pth.tar'))
        utils.load_checkpoint(os.path.join(model_name, 'best.pth.tar'), model)
        model.eval()
        print(os.path.join(model_name, 'best.pth.tar'))

        # predict
        test_batches = batching_list_instances(conf, tests)
        hard_constraint_predict(config=conf, model=model, fold_batches=test_batches, folded_insts=tests)

        for idx in range(len(tests)):
            prediction = tests[idx].output_ids
            prediction = [cfig.idx2labels[l] for l in prediction]
            # 测试一下标答labels的抽取
            # prediction = tests[idx].output
            tests[idx].prediction = prediction
        for idx in range(len(tests)):
            qids_original = tests[idx].id
            qids = re.search('\d+', qids_original).group()
            sub_id = re.search('\d+$', qids_original).group()
            # data_type = type_map[suffix]
            start = -1
            for i in range(len(tests[idx].prediction)):
                if tests[idx].prediction[i].startswith("B-") and start == -1:
                    start = i
                    # 找出单字实体(仅针对NUM类别)
                    # if tests[idx].prediction[i] == "B-NUM":
                    #     if i == len(tests[idx].prediction) - 1 or tests[idx].prediction[i + 1].startswith("B-") or \
                    #             tests[idx].prediction[i + 1].startswith("O"):
                    #         name = predict_dict['query'][i]
                    #         predict_dict[tests[idx].prediction[i][2:]] = predict_dict.get(tests[idx].prediction[i][2:],
                    #                                                                       []) + [
                    #                                                          {"str": name, "start_position": i,
                    #                                                           "end_position": i}]
                    #         start = -1
                if tests[idx].prediction[i].startswith("E-") and start != -1:
                    # if i != len(tests[idx].prediction) - 1 and predict_dict['query'][i+1] == '之':  # 修正模型对于TV类别预测的不恰当分割
                    #     continue
                    # else:
                    if tests[idx].prediction[i][2:] == tests[idx].prediction[start][
                                                       2:]:  # START 和 END 的类别必须保持一致，否则不能算实体，放弃抽取
                        end = i
                        value = tests[idx].content[start:end + 1]
                        role = attributes_type_map_inv[tests[idx].prediction[i][2:]]
                        sample = {"text_id": qids, "attributes": [
                            {"entity": value, "start": start + 500 * int(sub_id), "end": end + 500 * int(sub_id),
                             "type": role}
                        ]
                                  }
                        lst.append(sample)
                        start = -1
                    else:
                        start = -1
    sub_data = open(f'{conf.model_folder}/dev_prediction.txt', 'w+', encoding='utf-8')
    official_test_df = open('data/train_fix/ccks_task1_train.txt', 'r', encoding='utf-8').readlines()
    # official_test_transfer_df = open('data/dev/trans_dev.json', 'r', encoding='utf-8').readlines()
    # official_test_df.extend(official_test_transfer_df)
    merge_dict = dict()
    # 获取所有的id集合
    idx_list = []
    import json
    for line in tqdm(official_test_df):
        line = line.strip()
        line = json.loads(line)
        ids = line['text_id']
        idx_list.append(ids)
    for k in tqdm(lst):
        sam_dic = {"text_id": k['text_id'], "attributes": [k['attributes'][0]]}
        if k['text_id'] not in merge_dict.keys():
            merge_dict[k['text_id']] = sam_dic
        else:
            merge_dict[k['text_id']]["attributes"].append(k['attributes'][0])
        if k['text_id'] in idx_list:
            idx_list.remove(k['text_id'])
    merge_lst = list(merge_dict.values())
    for ids in tqdm(idx_list):
        merge_lst.append({"text_id": ids, "attributes": []})
    for i in tqdm(merge_lst):
        # ids = i['text_id']
        # events = i['attribute']
        # sub_dic = {}
        # info_dic = {}
        # for d in events:
        #     if d['type'] not in info_dic:
        #         info_dic[d['type']] = d['mentions']
        #         list1 = info_dic[d['type']]
        #     else:
        #         info_dic[d['type']] = info_dic[d['type']] + d['mentions']
        # sub_dic['text_id'] = ids
        # t_list = []
        # for key, value in info_dic.items():
        #     dic1 = {}
        #     dic1['type'] = key
        #     dic1['mentions'] = value
        #     t_list.append(dic1)
        # sub_dic['attribute'] = t_list
        # print(sub_dic)
        json.dump(i, sub_data, ensure_ascii=False)
        sub_data.write('\n')




def main_train_valid():
    logging.info("Transformer implementation")
    parser = argparse.ArgumentParser(description="Transformer CRF implementation")
    opt = parse_arguments_t(parser)
    conf = Config(opt)
    conf.train_file = conf.dataset + "/train.txt"
    conf.dev_file = conf.dataset + "/valid.txt"
    # conf.test_file = conf.dataset + "/valid.txt"
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device_num

    # data reader
    reader = Reader(conf.digit2zero)
    set_seed(opt, conf.seed)

    # read tests
    logging.info("\n")
    logging.info("Loading the datasets...")
    trains = reader.read_txt(conf.train_file, conf.train_num)
    devs = reader.read_txt(conf.dev_file, conf.dev_num)
    # tests = reader.read_txt(conf.test_file, conf.train_num)
    # query_list = reader.get_origin_query(conf.test_file, conf.train_num)
    # assert len(query_list) == len(tests)

    logging.info("Building label idx ...")
    # build label2idx and idx2label
    conf.build_label_idx(trains + devs)

    # load model
    cfig_path = os.path.join(conf.bert_model_dir, 'bert_config.json')
    cfig = BertConfig.from_json_file(cfig_path)
    cfig.device = conf.device
    cfig.label2idx = conf.label2idx
    cfig.label_size = conf.label_size
    cfig.idx2labels = conf.idx2labels
    model_folder = conf.model_folder
    model_name = model_folder + "/final_bert_crf"
    # model = BertCRF.from_pretrained(conf.bert_model_dir, config=cfig)
    model = BertCRF(cfig=cfig)
    model.to(cfig.device)
    utils.load_checkpoint(os.path.join(model_name, 'best.pth.tar'), model)
    if conf.full_finetuning:
        logging.info('full finetuning')
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]

    else:
        logging.info('tuning downstream layer')
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]

    optimizer = Adam(optimizer_grouped_parameters, lr=conf.learning_rate)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch))
    model.train()
    print(os.path.join(model_name, 'best.pth.tar'))
    # split devs into train, devs
    shuffle_idx = list(range(len(devs)))
    random.shuffle(shuffle_idx)
    trains = devs[:int(len(devs)*0.8)]
    devs = devs[int(len(devs)*0.8):]
    print('【trains: '+str(len(trains))+' devs: '+str(len(devs))+'】')
    trains_batches = batching_list_instances(conf, trains)
    devs_batches = batching_list_instances(conf, devs)
    valid_model_folder = model_folder + '_4-1_valid-9'
    if not os.path.exists(valid_model_folder):
        os.makedirs(valid_model_folder)
    valid_model_name = valid_model_folder + "/final_bert_crf"
    config_name = valid_model_folder + "/config.conf"

    # set logger
    utils.set_logger(os.path.join(valid_model_folder, 'train.log'))

    # params
    for k in opt.__dict__:
        logging.info(k + ": " + str(opt.__dict__[k]))
    epoch = conf.num_epochs
    best_dev_f1 = -1
    for i in range(1, epoch + 1):
        epoch_loss = 0
        start_time = time.time()
        model.zero_grad()

        for index in np.random.permutation(len(trains_batches)):  # disorder the train batches
            model.train()
            scheduler.step()
            input_ids, input_seq_lens, annotation_mask, labels = trains_batches[index]
            input_masks = input_ids.gt(0)
            # update loss
            loss = model(input_ids, input_seq_lens=input_seq_lens, annotation_mask=annotation_mask,
                         labels=labels, attention_mask=input_masks)
            epoch_loss += loss.item()
            model.zero_grad()
            loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=conf.clip_grad)
            optimizer.step()
        end_time = time.time()
        logging.info("Epoch %d: %.5f, Time is %.2fs" % (i, epoch_loss, end_time - start_time))

        model.eval()
        with torch.no_grad():
            # metric is [precision, recall, f_score]
            dev_metrics = evaluate_model(conf, model, devs_batches, "dev", devs)
            if dev_metrics[2] > best_dev_f1:  # save the best model
                logging.info("saving the best model...")
                best_dev_f1 = dev_metrics[2]

                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                optimizer_to_save = optimizer
                utils.save_checkpoint({'epoch': epoch + 1,
                                       'state_dict': model_to_save.state_dict(),
                                       'optim_dict': optimizer_to_save.state_dict()},
                                      is_best=dev_metrics[2] > 0,
                                      checkpoint=valid_model_name)

                # Save the corresponding config as well.
                if config_name:
                    f = open(config_name, 'wb')
                    pickle.dump(conf, f)
                    f.close()
        model.zero_grad()
    # load the best final model
    utils.load_checkpoint(os.path.join(valid_model_name, 'best.pth.tar'), model)
    model.eval()
    logging.info("\n")
    result = evaluate_model(conf, model, devs_batches, "dev", devs)
    logging.info("\n\n")


def main_predict_voting():
    logging.info("Transformer implementation")
    parser = argparse.ArgumentParser(description="Transformer CRF implementation")
    opt = parse_arguments_t(parser)
    conf = Config(opt)
    conf.train_file = conf.dataset + "/train.txt"
    conf.dev_file = conf.dataset + "/valid.txt"
    conf.test_file = conf.dataset + "/test.txt"
    # os.environ['CUDA_VISIBLE_DEVICES'] = opt.device_num

    # data reader
    reader = Reader(conf.digit2zero)

    # read tests
    logging.info("\n")
    logging.info("Loading the datasets...")
    trains = reader.read_txt(conf.train_file, conf.train_num)
    devs = reader.read_txt(conf.dev_file, conf.dev_num)
    tests = reader.read_txt(conf.test_file, conf.train_num)
    query_list = reader.get_origin_query(conf.test_file, conf.train_num)
    assert len(query_list) == len(tests)

    logging.info("Building label idx ...")
    # build label2idx and idx2label
    conf.build_label_idx(trains + devs)

    # load model
    cfig_path = os.path.join(conf.bert_model_dir, 'bert_config.json')
    cfig = BertConfig.from_json_file(cfig_path)
    cfig.device = conf.device
    cfig.label2idx = conf.label2idx
    cfig.label_size = conf.label_size
    cfig.idx2labels = conf.idx2labels
    tests_list = []
    # voting
    for i in range(10):
        tests_temp = copy.deepcopy(tests)
        model_folder = conf.model_folder + str(i)
        model_name = model_folder + "/final_bert_crf"
        # model = BertCRF.from_pretrained(conf.bert_model_dir, config=cfig)
        model = BertCRF(cfig=cfig)
        model.to(cfig.device)
        utils.load_checkpoint(os.path.join(model_name, 'best.pth.tar'), model)
        model.eval()
        print(os.path.join(model_name, 'best.pth.tar'))

        # predict
        test_batches = batching_list_instances(conf, tests_temp)
        hard_constraint_predict(config=conf, model=model, fold_batches=test_batches, folded_insts=tests_temp)
        tests_list.append(tests_temp)
    print("VOTING...")
    for idx in tqdm(range(len(tests))):
        output_ids = []
        for i in range(len(tests_list[0][idx].output_ids)):
            vote_list = [t[idx].output_ids[i] for t in tests_list]
            vote_result = max(vote_list, key=vote_list.count)
            output_ids.append(vote_result)
        prediction = output_ids
        prediction = [cfig.idx2labels[l] for l in prediction]
        # 测试一下标答labels的抽取
        # prediction = tests[idx].output
        tests[idx].prediction = prediction
    with open('../submission/YourTeamName1.json', 'w+', encoding='utf-8') as file:
        for idx in range(len(tests)):
            predict_dict = dict()
            predict_dict['id'] = '0' * (5-len(str(idx+1))) + str(idx+1)
            # predict_dict['query'] = ''.join(tests[idx].input.get_words())
            predict_dict['query'] = query_list[idx]

            # q全角半角空格处理
            # if '&' in predict_dict['query']:
            #     query = ''
            #     for ch_id, ch in enumerate(predict_dict['query']):
            #         if ch == '&':
            #             if predict_dict['query'][ch_id-1].encode().isdigit() or predict_dict['query'][ch_id+1].encode().isdigit() or predict_dict['query'][ch_id-1].encode().isalpha() or predict_dict['query'][ch_id+1].encode().isalpha():
            #                 query += " "
            #             else:
            #                 query += "　"
            #         else:
            #             query += ch
            #     predict_dict['query'] = query
            start = -1
            for i in range(len(tests[idx].prediction)):
                if tests[idx].prediction[i].startswith("B-") and start == -1:
                    start = i
                    # 找出单字实体(仅针对NUM类别)
                    if tests[idx].prediction[i] == "B-NUM":
                        if i == len(tests[idx].prediction) - 1 or tests[idx].prediction[i + 1].startswith("B-") or tests[idx].prediction[i + 1].startswith("O"):
                            name = predict_dict['query'][i]
                            predict_dict[tests[idx].prediction[i][2:]] = predict_dict.get(tests[idx].prediction[i][2:], []) + [{"str": name, "start_position": i, "end_position": i}]
                            start = -1
                if tests[idx].prediction[i].startswith("E-") and start != -1:
                    # if i != len(tests[idx].prediction) - 1 and predict_dict['query'][i+1] == '之':  # 修正模型对于TV类别预测的不恰当分割
                    #     continue
                    # else:
                    if tests[idx].prediction[i][2:] == tests[idx].prediction[start][2:]:  # START 和 END 的类别必须保持一致，否则不能算实体，放弃抽取
                        end = i
                        if i == len(tests[idx].prediction) - 1:
                            name = predict_dict['query'][start:]
                        else:
                            name = predict_dict['query'][start:end+1]
                        predict_dict[tests[idx].prediction[i][2:]] = predict_dict.get(tests[idx].prediction[i][2:], []) + [
                            {"str": name, "start_position": start, "end_position": end}]
                        start = -1
                    else:
                        start = -1
            import json
            jsobj = json.dumps(predict_dict, ensure_ascii=False)
            file.write(jsobj+'\n')


def main_stacking():
    logging.info("Transformer implementation")
    # parser = argparse.ArgumentParser(description="Transformer CRF implementation")
    # opt = parse_arguments_t(parser)
    conf = Config(opt)
    conf.train_file = conf.dataset + "/train_fix"
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device_num
    # data reader
    reader = Reader(conf.digit2zero)
    set_seed(opt, conf.seed)

    if not os.path.exists(conf.model_folder):
        os.makedirs(conf.model_folder)

    # set logger
    utils.set_logger(os.path.join(conf.model_folder, 'train.log'))

    # params
    for k in opt.__dict__:
        logging.info(k + ": " + str(opt.__dict__[k]))

    # read trains/devs
    logging.info("\n")
    logging.info("Loading the datasets...")
    trains_add_devs = reader.read_txt(conf.train_file, conf.train_num, opt.type, int(conf.max_len) - 12)

    logging.info("Building label idx ...")
    # build label2idx and idx2label
    conf.build_label_idx(list(chain.from_iterable(trains_add_devs)))

    if opt.shuffle_reading:
        logging.info("\n")
        logging.info(f"Shuffle reading the datasets...Seed:{conf.seed}")
        random.shuffle(trains_add_devs)
    trains = list(chain.from_iterable(trains_add_devs[:int(opt.train_dev_split_rate * len(trains_add_devs))]))
    devs = list(chain.from_iterable(trains_add_devs[int(opt.train_dev_split_rate * len(trains_add_devs)):]))
    if opt.throw_all_O_sample:
        trains = [t for t in trains if len(t.mentions) != 0]  # 去掉训练集全O的样本
        # devs = [d for d in devs if len(d.mentions) != 0]  # 去掉开发集全O的样本
    print('【trains: ' + str(len(trains)) + ' devs: ' + str(len(devs)) + '】')
    random.shuffle(trains)
    # set the prediction flag, if is_prediction is False, we will not update this label.
    for inst in trains:
        inst.is_prediction = [False] * len(inst.input)
        for pos, label in enumerate(inst.output):
            if label == conf.O:
                inst.is_prediction[pos] = True
    # dividing the data into 2 parts(num_folds default to 2)
    num_insts_in_fold = math.ceil(len(trains) / conf.num_folds)
    trains = [trains[i * num_insts_in_fold: (i + 1) * num_insts_in_fold] for i in range(conf.num_folds)]

    train_model(config=conf, train_insts=trains, dev_insts=devs)


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = opt.device
    parser = argparse.ArgumentParser(description="Transformer CRF implementation")
    opt = parse_arguments_t(parser)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device_num
    print(torch.cuda.current_device())
    if opt.train_or_predict == 1:  # 模型训练
        main()
    elif opt.train_or_predict == 2:  # 模型预测
        main_predict()
    else:
        logging.info("Wrong mode!")
