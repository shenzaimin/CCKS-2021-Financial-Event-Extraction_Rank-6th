# -*- coding: utf-8 -*-
# @Time     : 2021/6/19 13:55
# @Author   : 宁星星
# @Email    : shenzimin0@gmail.com
import os
import json
import re
import copy
from label_error_detect import label_error_detect


file_dir = '../data/train'
fix_file_dir = '../data_complete/train/ccks_task1_train_3.txt'
error_dict = label_error_detect()
print(error_dict)
for f_temp in os.listdir(file_dir):
    mention_add_num = 0
    file = os.path.join(file_dir, f_temp)
    print("Reading file: " + file)
    in_file = open(file, 'r', encoding='utf-8')
    out_file = open(fix_file_dir, 'w+', encoding='utf-8')
    for line in in_file:
        line = line.strip()
        line = json.loads(line)
        words_original = line['text']
        idx_original = line['text_id']
        new_attributes = copy.deepcopy(line['attributes'])
        for mention in line['attributes']:
            role = mention['type']
            entity = mention['entity']
            if role != '资损金额' and role != '支付渠道' and role != '案发时间' and "*" not in entity and ")" not in entity and "?" not in entity:
                if idx_original in error_dict:
                    if entity in error_dict[idx_original]:
                        print(f'{idx_original}-{role}-{error_dict[idx_original]}')
                        continue
                entity_span_list = [match.span() for match in list(re.finditer(entity, words_original))]
                for span in entity_span_list:
                    start = span[0]
                    end = span[1] - 1
                    mention_new = {'start': start, 'type': role, 'end': end, 'entity': entity}
                    if mention_new not in new_attributes:
                        new_attributes.append(mention_new)
                        mention_add_num += 1
        line['attributes'] = new_attributes
        json_obj = json.dumps(line, ensure_ascii=False)
        out_file.write(json_obj + '\n')
    print(f'Mention added {mention_add_num}')



file_dir = '../data_complete/train'
fix_file_dir = '../data_complete/train_fix/ccks_task1_train_2.txt'
# os.mkdir('../data/train_fix')
for f_temp in os.listdir(file_dir):
    wrong = 0
    file = os.path.join(file_dir, f_temp)
    print("Reading file: " + file)
    in_file = open(file, 'r', encoding='utf-8')
    out_file = open(fix_file_dir, 'w+', encoding='utf-8')
    for line in in_file:
        line = line.strip()
        line = json.loads(line)
        words_original = line['text']
        idx_original = line['text_id']
        new_attributes = copy.deepcopy(line['attributes'])
        for mention in line['attributes']:
            start_span = mention['start']
            end_span = mention['end'] + 1
            role = mention['type']
            entity = mention['entity']
            # check有多少标错的实体
            if entity != words_original[start_span:end_span]:
                new_mention_left = {}
                new_mention_right = {}
                # check left
                slide_dist_left = 0
                start_span_left = mention['start']
                end_span_left = mention['end'] + 1
                while entity != words_original[start_span_left:end_span_left] and slide_dist_left < 70 and start_span_left >= 0:
                    start_span_left -= 1
                    end_span_left -= 1
                    slide_dist_left += 1

                if entity != words_original[start_span_left:end_span_left]:  # 如果向左没有匹配上
                    # check right
                    slide_dist_right = 0
                    start_span_right = mention['start']
                    end_span_right = mention['end'] + 1
                    while entity != words_original[
                                         start_span_right:end_span_right] and slide_dist_right < 70:
                        start_span_right += 1
                        end_span_right += 1
                        slide_dist_right += 1
                    if entity != words_original[start_span_right:end_span_right]:  # 如果向右也没有匹配上，则交换匹配值和标注答案
                        start_span = mention['start']
                        end_span = mention['end'] + 1
                        print(
                            f'{idx_original}-[WRONG]-{words_original[start_span:end_span]}-[right]-{mention["entity"]}-{start_span}')
                        wrong += 1
                        new_attributes.remove(mention)
                        new_mention = {'start': start_span, 'type': mention['type'],
                                       'end': end_span - 1, 'entity': words_original[start_span:end_span]}
                        new_attributes.append(new_mention)
                    else:
                        new_attributes.remove(mention)
                        new_mention = {'start': start_span_right, 'type': mention['type'],
                                       'end': end_span_right - 1, 'entity': mention['entity']}
                        new_attributes.append(new_mention)
                        print(f'                                            right slide distance -->>: {slide_dist_right}')
                        if slide_dist_right > 60:
                            print(idx_original)
                else:
                    new_attributes.remove(mention)
                    new_mention_left = {'start': start_span_left, 'type': mention['type'],
                                   'end': end_span_left - 1, 'entity': mention['entity']}

                    # check right
                    slide_dist_right = 0
                    start_span_right = mention['start']
                    end_span_right = mention['end'] + 1
                    while entity != words_original[
                                    start_span_right:end_span_right] and slide_dist_right < 70:
                        start_span_right += 1
                        end_span_right += 1
                        slide_dist_right += 1
                    if entity != words_original[start_span_right:end_span_right]:  # 如果向右没有匹配上，则取左边匹配结果，否则比较偏移值（取较小的）
                        new_attributes.append(new_mention_left)
                        print(f'<<-- left slide distance: {slide_dist_left}')
                    else:
                        new_mention_right = {'start': start_span_right, 'type': mention['type'],
                                       'end': end_span_right - 1, 'entity': mention['entity']}
                        if slide_dist_right <= slide_dist_left:
                            new_attributes.append(new_mention_right)
                            print(f'                                            right slide distance -->>: {slide_dist_right}')
                            if slide_dist_right > 60:
                                print(idx_original)
                        else:
                            new_attributes.append(new_mention_left)
                            print(f'<<-- left slide distance: {slide_dist_left}')

        line['attributes'] = new_attributes
        json_obj = json.dumps(line, ensure_ascii=False)
        out_file.write(json_obj+'\n')