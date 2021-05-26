# -*- coding: utf-8 -*-
# @Time     : 2021/5/25 17:46
# @Author   : 宁星星
# @Email    : shenzimin0@gmail.com
import json


third_submit = open('../验证集提交/第三次/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8').readlines()
fifth_submit = open('../验证集提交/第五次/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8').readlines()

sixth_submit = open('../验证集提交/第六次/系统之神与我同在_valid_result.txt', 'w+', encoding='utf-8')

third_dict = dict()
for line in third_submit:
    line = line.strip()
    line = json.loads(line)
    ids = line['text_id']
    attributes = line["attributes"]
    third_dict[ids] = attributes

for line in fifth_submit:
    line = line.strip()
    line = json.loads(line)
    ids = line['text_id']
    attributes_fifth = line["attributes"]
    attributes_third = third_dict[ids]
    new_line = dict()
    new_line["text_id"] = ids
    new_attributes = []
    entity_words = []
    for att in attributes_fifth:
        new_attributes.append(att)
        entity_words.append(att['entity'])

    for att_third in attributes_third:
        if att_third["entity"] not in entity_words:
            new_attributes.append(att_third)
    new_line["attributes"] = new_attributes
    json.dump(new_line, sixth_submit, ensure_ascii=False)
    sixth_submit.write('\n')
