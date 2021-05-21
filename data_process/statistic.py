# -*- coding: utf-8 -*-
# @Time     : 2021/5/20 14:45
# @Author   : 宁星星
# @Email    : shenzimin0@gmail.com
import jsonlines
import pandas as pd


def count_event_types(data_dir):
    data_num = 0
    event_set = set()
    with open(data_dir, 'r', encoding='utf8') as file:
        for item in jsonlines.Reader(file):
            data_num += 1
            event_set.add(item['level1'])
            event_set.add(item['level2'])
            event_set.add(item['level3'])
    print(f'data_num: {data_num}')
    print(f'event_num: {len(event_set)}')
    print(f'all_event_type: {event_set}')
    return event_set


def get_event_schema(data_dir, event_set):
    schema = dict()
    all_types = set()
    with open(data_dir, 'r', encoding='utf8') as file:
        for item in jsonlines.Reader(file):
            attributes = item['attributes']
            type_set = set()
            for att in attributes:
                type_set.add(att['type'])
            all_types = all_types.union(type_set)
            schema[item['level1']] = schema.get(item['level1'], set()).union(type_set)

            schema[item['level2']] = schema.get(item['level2'], set()).union(type_set)

            schema[item['level3']] = schema.get(item['level3'], set()).union(type_set)
        for k, v in schema.items():
            schema[k] = list(v)
    print(f'all types: {all_types}')
    return schema


if __name__ == '__main__':
    data_dir_train = '../data/train/ccks_task1_train.txt'
    data_dir_dev = '../data/dev/ccks_task1_eval_data.txt'
    train_event = count_event_types(data_dir_train)
    print('*'*50)
    dev_event = count_event_types(data_dir_dev)
    print('*'*50)
    print(f'Common event types: {len(train_event.intersection(dev_event))}')
    schema = get_event_schema(data_dir_train, train_event)
    print(schema)