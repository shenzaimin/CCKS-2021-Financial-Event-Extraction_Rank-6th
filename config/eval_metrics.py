import json
import pandas as pd
import numpy as np


class Span:
    """
    A class of `Span` where we use it during evaluation.
    We construct spans for the convenience of evaluation.
    """
    def __init__(self, left: int, right: int, type: str):
        """
        A span compose of left, right (inclusive) and its entity label.
        :param left:
        :param right: inclusive.
        :param type:
        """
        self.left = left
        self.right = right
        self.type = type

    def __eq__(self, other):
        return self.left == other.left and self.right == other.right and self.type == other.type

    def __hash__(self):
        return hash((self.left, self.right, self.type))

def error_output_csv_2(file_pre, file_true, output_dir):
    error_list = []
    content_dict = dict()
    pre_dict = dict()
    true_dict = dict()
    for line in open(file_pre,'r'):
        line = line.strip()
        line = json.loads(line)
        ids = line['id']
        pre_dict[ids] = line['events']
        content_dict[ids] = line['content']
    for line in open(file_true,'r'):
        line = line.strip()
        line = json.loads(line)
        ids = line['id']
        true_dict[ids] = line['events']

    for pre_id in pre_dict.keys():
        ids = pre_id
        content = content_dict[ids]
        pre_events =  pre_dict[pre_id]
        for evn_pre in pre_events:
            type = evn_pre['type']
            true_events = true_dict[pre_id]
            mentions_pre = dict()
            for m in evn_pre['mentions']:
                mentions_pre[m['role']] = m['word']
            mentions_true = dict()
            for evn_true in true_events:
                if evn_true['type'] == type:
                    # mentions_true = dict()
                    for m in evn_true['mentions']:
                        if not m['role'] in mentions_true:
                            mentions_true[m['role']] = set([m['word']])
                        else:
                            mentions_true[m['role']].add(m['word'])
            for k, v in mentions_true.items():
                if k in mentions_pre.keys():
                    if mentions_pre[k] not in v:
                        error_list.append([ids, type, k, list(v)[0], mentions_pre[k], content])
                    else:
                        v.remove(mentions_pre[k])
                        for v_ in v:
                            error_list.append([ids, type, k, v_, '', content])
                elif k not in mentions_pre.keys():
                    for v_ in v:
                        error_list.append([ids, type, k, v_, '', content])
    error_array = np.asarray(error_list)
    df = pd.DataFrame(error_array, index=None)
    df.columns = ['id', 'type', 'role', 'answer', 'predict', 'content']
    df.to_csv(output_dir, index=False)

def error_output_csv_3(file_pre, file_true, output_dir):
    error_list = []
    content_dict = dict()
    pre_dict = dict()
    true_dict = dict()
    for line in open(file_pre,'r'):
        line = line.strip()
        line = json.loads(line)
        ids = line['id']
        pre_dict[ids] = line['events']
        content_dict[ids] = line['content']
    for line in open(file_true,'r'):
        line = line.strip()
        line = json.loads(line)
        ids = line['id']
        true_dict[ids] = line['events']

    for pre_id in pre_dict.keys():
        ids = pre_id
        content = content_dict[ids]
        pre_events =  pre_dict[pre_id]
        for evn_pre in pre_events:
            type = evn_pre['type']
            true_events = true_dict[pre_id]
            mentions_pre = dict()
            for m in evn_pre['mentions']:
                if not m['role'] in mentions_pre:
                    mentions_pre[m['role']] = set([m['word']])
                else:
                    mentions_pre[m['role']].add(m['word'])
                # mentions_pre[m['role']] = m['word']
            mentions_true = dict()
            for evn_true in true_events:
                if evn_true['type'] == type:
                    # mentions_true = dict()
                    for m in evn_true['mentions']:
                        if not m['role'] in mentions_true:
                            mentions_true[m['role']] = set([m['word']])
                        else:
                            mentions_true[m['role']].add(m['word'])
            for k, v in mentions_true.items():
                if k in mentions_pre.keys():
                    if len(v) >1:
                        for v_ in v:
                            if v_ not in mentions_pre[k]:
                                error_list.append([ids, type, k, v_, '', content])
                    else:
                        for p in mentions_pre[k]:
                            if p not in v:
                                error_list.append([ids, type, k, list(v)[0], p, content])
                    # if mentions_pre[k] not in v:
                    #     error_list.append([ids, type, k, list(v)[0], mentions_pre[k], content])
                    # else:
                    #     v.remove(mentions_pre[k])
                    #     for v_ in v:
                    #         error_list.append([ids, type, k, v_, '', content])
                elif k not in mentions_pre.keys():
                    for v_ in v:
                        error_list.append([ids, type, k, v_, '', content])
    error_array = np.asarray(error_list)
    df = pd.DataFrame(error_array, index=None)
    df.columns = ['id', 'type', 'role', 'answer', 'predict', 'content']
    df.to_csv(output_dir, index=False)

def evaluate(file_pre, file_true):
    type_dict = {'质押': [
        'trigger',
        'sub-org',
        'sub-per',
        'obj-org',
        'obj-per',
        'collateral',
        'date',
        'money',
        'number',
        'proportion'
    ],
        '股份股权转让':
            [
                'trigger',
                'sub-org',
                'sub-per',
                'obj-org',
                'obj-per',
                'collateral',
                'date',
                'money',
                'number',
                'proportion',
                'target-company'
            ],
        '起诉':
            [
                'trigger',
                'sub-org',
                'sub-per',
                'obj-org',
                'obj-per',
                'date',
            ],
        '投资':
            [
                'trigger',
                'sub',
                'obj',
                'money',
                'date',
            ],
        '减持':
            [
                'trigger',
                'sub',
                'title',
                'date',
                'share-per',
                'share-org',
                'obj',
            ],
        '收购':
            [
                'trigger',
                'sub-org',
                'sub-per',
                'obj-org',
                'way',
                'date',
                'money',
                'number',
                'proportion',
            ],
        '判决':
            [
                'trigger',
                'sub-per',
                'sub-org',
                'institution',
                'obj-per',
                'obj-org',
                'date',
                'money',
            ]}
    pre_dict = dict()
    true_dict = dict()
    for line in open(file_pre, 'r'):
        line = line.strip()
        line = json.loads(line)
        ids = line['id']
        pre_dict[ids] = line['events']
    for line in open(file_true, 'r'):
        line = line.strip()
        line = json.loads(line)
        ids = line['id']
        true_dict[ids] = line['events']
    total_metrics = np.zeros([3], dtype=int)
    for type in type_dict.keys():
        p_list = [0] * len(type_dict[type])
        total_entity_list = [0] * len(type_dict[type])
        total_predict_list = [0] * len(type_dict[type])
        for pre_id in pre_dict.keys():
            output_spans_list = []
            for i in range(len(type_dict[type])):
                output_spans_list.append(set())
            predict_spans_list = []
            for i in range(len(type_dict[type])):
                predict_spans_list.append(set())
            pre_events = pre_dict[pre_id]
            for evn_pre in pre_events:
                if evn_pre['type'] == type:
                    true_events = true_dict[pre_id]
                    for mention_pre in evn_pre['mentions']:
                        start = mention_pre['span'][0]
                        end = mention_pre['span'][1]
                        role = mention_pre['role']
                        predict_spans_list[type_dict[type].index(role)].add(Span(start, end, role))
                    for evn_true in true_events:
                        if evn_true['type'] == type:
                            for mention_true in evn_true['mentions']:
                                start = mention_true['span'][0]
                                end = mention_true['span'][1]
                                role = mention_true['role']
                                output_spans_list[type_dict[type].index(role)].add(Span(start, end, role))
            for j in range(len(type_dict[type])):
                total_entity_list[j] += len(output_spans_list[j])
            for k in range(len(type_dict[type])):
                total_predict_list[k] += len(predict_spans_list[k])
            for m in range(len(type_dict[type])):
                p_list[m] += len(predict_spans_list[m].intersection(output_spans_list[m]))
        metrics = np.asarray([p_list, total_predict_list, total_entity_list], dtype=int).transpose()
        all_metircs = metrics.sum(axis=0)
        total_metrics +=all_metircs
        p, total_predict, total_entity = all_metircs[0], all_metircs[1], all_metircs[2]
        precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
        recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
        fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
        print('*'*50)
        print("[* %s *] Precision: %.2f, Recall: %.2f, F1: %.2f" % (type, precision, recall, fscore))
        for i in range(len(type_dict[type])):
            p, total_predict, total_entity = metrics[i][0], metrics[i][1], metrics[i][2]
            # calculate the precision, recall and f1 score
            # p, total_predict, total_entity = metrics[0], metrics[1], metrics[2]
            precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
            recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
            fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
            print("[* %s *] Precision: %.2f, Recall: %.2f, F1: %.2f" % (type_dict[type][i], precision, recall, fscore))
    p, total_predict, total_entity = total_metrics[0], total_metrics[1], total_metrics[2]
    precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
    recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
    fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    print('*' * 50)
    print("[* 汇总 *] Precision: %.2f, Recall: %.2f, F1: %.2f" % (precision, recall, fscore))
    print('*' * 50)




if __name__ == "__main__":
    file_pre = '../data/error/train_valid_result.json'
    file_true = '../data/error/train/valid_true.json'
    output_dir = '../data/error/error_info.csv'
    error_output_csv_3(file_pre, file_true, output_dir)
    # evaluate(file_pre, file_true)