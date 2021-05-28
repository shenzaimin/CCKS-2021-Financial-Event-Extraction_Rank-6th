# -*- coding: utf-8 -*-
# @Time     : 2021/5/25 17:46
# @Author   : 宁星星
# @Email    : shenzimin0@gmail.com
import json
import re


def trick_1(input_1, input_2, output):
    entity_num = 0
    replace_dict = dict()
    ref_dict = dict()
    for line in input_1:
        line = line.strip()
        line = json.loads(line)
        ids = line['text_id']
        attributes = line["attributes"]
        ref_dict[ids] = attributes

    for line in input_2:
        line = line.strip()
        line = json.loads(line)
        ids = line['text_id']
        attributes_new = line["attributes"]
        attributes_ref = ref_dict[ids]
        new_line = dict()
        new_line["text_id"] = ids
        new_attributes = []
        entity_words = []
        for att in attributes_new:
            new_attributes.append(att)
            entity_words.append(att['entity'])

        for att_ref in attributes_ref:
            if att_ref["entity"] not in entity_words:
                # print(att_third["type"])
                replace_dict[att_ref["type"]] = replace_dict.get(att_ref["type"], 0) + 1
                new_attributes.append(att_ref)
        new_line["attributes"] = new_attributes
        entity_num += len(new_attributes)
        json.dump(new_line, output, ensure_ascii=False)
        output.write('\n')
    print(replace_dict)
    print(f'predicted entity num: {entity_num}')


def trick_2(input_1, input_2, output, rep_entity_list):
    """
    只对准确率比较高的实体进行补全：
    资损金额（Precision: 66.67, Recall: 78.04, F1: 71.91）
    支付渠道（Precision: 55.94, Recall: 60.75, F1: 58.25）
    涉案平台（Precision: 53.12, Recall: 54.26, F1: 53.68）
    """
    third_dict = dict()
    for line in input_1:
        line = line.strip()
        line = json.loads(line)
        ids = line['text_id']
        attributes = line["attributes"]
        third_dict[ids] = attributes

    for line in input_2:
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
                # print(att_third["type"])
                if att_third["type"] in rep_entity_list:
                    new_attributes.append(att_third)
                    print(att_third["type"])
        new_line["attributes"] = new_attributes
        json.dump(new_line, output, ensure_ascii=False)
        output.write('\n')


def extract_sfzh_by_reg(origin_file, inp, output):
    pattern = r"身份证号码(\d+\**)"
    ref_dict = dict()
    for line in origin_file:
        line = line.strip()
        line = json.loads(line)
        ids = line['text_id']
        text = line["text"]
        ref_dict[ids] = text
    for line in inp:
        line = line.strip()
        line = json.loads(line)
        ids = line['text_id']
        attributes = line['attributes']
        text = ref_dict[ids]
        if re.search(pattern, text):
            new_att = {"entity": re.search(pattern, text).group(1), "start": re.search(pattern, text).regs[1][0], "end": re.search(pattern, text).regs[1][1]-1, "type": "身份证号"}
            attributes.append(new_att)
            print(new_att)
        json.dump(line, output, ensure_ascii=False)
        output.write('\n')
    # return re.search(pattern, text).group(1)


def extract_sfzh_by_reg_tmp(text):
    pattern = r"身份证号码(\d+\**)"

    return re.search(pattern, text).group(1)

if __name__ == '__main__':
    # third_submit = open('../验证集提交/第三次/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8').readlines()
    # fifth_submit = open('../验证集提交/第五次/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8').readlines()
    # sixth_submit = open('../验证集提交/第六次/系统之神与我同在_valid_result.txt', 'w+', encoding='utf-8')  # 合并第三次和第五次的结果，得到第六次的结果
    # trick_1(third_submit, fifth_submit, sixth_submit)

    third_submit = open('../验证集提交/第三次/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8').readlines()
    tenth_submit = open('../验证集提交/第十次（修正数据重跑第五次）/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8').readlines()
    eleventh_submit = open('../验证集提交/第十一次/系统之神与我同在_valid_result.txt', 'w+', encoding='utf-8')  # 合并第三次和第五次的结果，得到第六次的结果
    trick_1(third_submit, tenth_submit, eleventh_submit)

    # valid_file_dir = open('../data/dev/ccks_task1_eval_data.txt', 'r', encoding='utf-8')
    # sixth_submit_r = open('../验证集提交/第六次/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8')
    # ninth_submit = open('../验证集提交/第九次/系统之神与我同在_valid_result.txt', 'w+', encoding='utf-8')
    # extract_sfzh_by_reg(valid_file_dir, sixth_submit_r, ninth_submit)
    # print(extract_sfzh_by_reg_tmp("长兴县人民检察院 起 诉 书 长检一部刑诉〔2020〕64号 被告人杨某甲,男,1998年**月**日出生,居民身份证号码3305221998********,汉族,初中文化程度,住浙江省长兴县**镇**村**自然村**号,长兴县"))

    # seventh_submit = open('../验证集提交/第七次/系统之神与我同在_valid_result.txt', 'w+', encoding='utf-8')
    # trick_2(third_submit, fifth_submit, seventh_submit, ["资损金额"])



    # fourth_submit = open('../验证集提交/第四次/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8').readlines()
    # sixth_submit_r = open('../验证集提交/第六次/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8')  # 合并第三次和第五次的结果，得到第六次的结果
    # eighth_submit = open('../验证集提交/第八次/系统之神与我同在_valid_result.txt', 'w+', encoding='utf-8')  # 继续合并第六次和第四次的结果
    # trick_1(fourth_submit, sixth_submit_r, eighth_submit)

