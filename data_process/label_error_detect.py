# -*- coding: utf-8 -*-
# @Time     : 2021/6/23 16:24
# @Author   : 宁星星
# @Email    : shenzimin0@gmail.com
import os
import json


def label_error_detect():
    file_dir = '../data/train'

    contradiction_dict = dict()
    for f_temp in os.listdir(file_dir):
        file = os.path.join(file_dir, f_temp)
        print("Reading file: " + file)
        in_file = open(file, 'r', encoding='utf-8')
        for line in in_file:
            line = line.strip()
            line = json.loads(line)
            words_original = line['text']
            idx_original = line['text_id']
            # attributes = line['attributes']
            xyr = set()
            shr =set()
            for mention in line['attributes']:
                role = mention['type']
                entity = mention['entity']
                if role == '受害人':
                    shr.add(entity)
                elif role == '嫌疑人':
                    xyr.add(entity)
            if len(xyr.intersection(shr)):
                contradiction_dict[idx_original] = contradiction_dict.get(idx_original, []) + list(xyr.intersection(shr))
                # print(f'{idx_original}-{xyr.intersection(shr)}')
        # print(contradiction_dict)
    return contradiction_dict

if __name__ == '__main__':
    print(label_error_detect())