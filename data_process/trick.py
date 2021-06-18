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


def get_rid_of_long_entity(inp, output):
    """
    案发时间除外
    """
    import string

    punctuation = """-。,！？｡＂"＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"""
    re_punctuation = "[{}]+".format(punctuation)
    for line in inp:
        new_line = dict()
        line = line.strip()
        line = json.loads(line)
        ids = line['text_id']
        attributes = line["attributes"]
        new_line['text_id'] = ids
        new_attributes = line["attributes"]
        for att in attributes:
            if re.search(re_punctuation, att['entity']) and att['type'] != "案发时间" and not att['entity'].endswith('元'):
                print(att['entity'])
                new_attributes.remove(att)
        new_line['attributes'] = new_attributes
        json.dump(line, output, ensure_ascii=False)
        output.write('\n')


def duplicate_entity(inp, output):
    """
    去除掉名称相同，但位置不同的“重复”实体, 用于验证官方评测有没有考虑span的坐标位置
    测试结果证明：完全没影响，不需要去重。
    """
    dup_num = 0
    for line in inp:
        attributes_list = []
        new_line = dict()
        line = line.strip()
        line = json.loads(line)
        ids = line['text_id']
        attributes = line["attributes"]
        new_line['text_id'] = ids
        new_attributes = line["attributes"]
        for att in attributes:
            att_new = {"entity": att['entity'], "type": att['type']}
            if att_new in attributes_list:
                new_attributes.remove(att)
                dup_num += 1
            else:
                attributes_list.append(att_new)
        new_line['attributes'] = new_attributes
        json.dump(line, output, ensure_ascii=False)
        output.write('\n')
        # if len(attributes_list):
        #     print(attributes_list)
    print(f'去重实体数目： {dup_num}个')


def add_text2submit(origin_file, inp, output):
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
        line['text'] = text
        json.dump(line, output, ensure_ascii=False)
        output.write('\n')


if __name__ == '__main__':
    # third_submit = open('../验证集提交/第三次/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8').readlines()
    # fifth_submit = open('../验证集提交/第五次/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8').readlines()
    # sixth_submit = open('../验证集提交/第六次/系统之神与我同在_valid_result.txt', 'w+', encoding='utf-8')  # 合并第三次和第五次的结果，得到第六次的结果
    # trick_1(third_submit, fifth_submit, sixth_submit)

    # third_submit = open('../验证集提交/第三次/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8').readlines()
    # tenth_submit = open('../验证集提交/第十次（修正数据重跑第五次）/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8').readlines()
    # eleventh_submit = open('../验证集提交/第十一次/系统之神与我同在_valid_result.txt', 'w+', encoding='utf-8')  # 合并第三次和第十次的结果，得到第十一次的结果
    # trick_1(third_submit, tenth_submit, eleventh_submit)

    # eleventh_submit = open('../验证集提交/第十一次/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8')
    # fifth_submit = open('../验证集提交/第五次/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8').readlines()
    # twelfth_submit = open('../验证集提交/第十二次/系统之神与我同在_valid_result.txt', 'w+', encoding='utf-8')
    # trick_1(fifth_submit, eleventh_submit, twelfth_submit)

    # third_submit = open('../验证集提交/第三次/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8').readlines()
    # thirteen_submit = open('../验证集提交/第十三次（数据按整个样本进行不切断划分训练，对比第五次）/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8')
    # fourteen_submit = open('../验证集提交/第十四次（融合第三次和第十三次）/系统之神与我同在_valid_result.txt', 'w+',
    #                       encoding='utf-8')
    # trick_1(third_submit, thirteen_submit, fourteen_submit)

    #fifth_submit = open('../验证集提交/第五次/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8').readlines()
    #thirteen_submit = open('../验证集提交/第十三次（数据按整个样本进行不切断划分训练，对比第五次）/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8')
   # fifteen_submit = open('../验证集提交/第十五次（融合第五次和第十三次）/系统之神与我同在_valid_result.txt', 'w+',
     #                      encoding='utf-8')
    #trick_1(fifth_submit, thirteen_submit, fifteen_submit)

   # third_submit = open('../验证集提交/第三次/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8').readlines()
   # sixteen_submit = open('../验证集提交/第十六次（前五模型集成）/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8')
   # seventeen_submit = open('../验证集提交/第十七次（合并16和3）/系统之神与我同在_valid_result.txt', 'w+',
  #  encoding = 'utf-8')
    #trick_1(third_submit, sixteen_submit, seventeen_submit)

    # thirteen_submit = open('../验证集提交/第十三次（数据按整个样本进行不切断划分训练，对比第五次）/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8').readlines()
    # seventeen_submit = open('../验证集提交/第十七次（合并16和3）/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8')
    # eighteen_submit = open('../验证集提交/第十八次（合并17和13）/系统之神与我同在_valid_result.txt', 'w+',
    #                         encoding='utf-8')
    # trick_1(thirteen_submit, seventeen_submit, eighteen_submit)

    # third_submit = open('../验证集提交/第三次/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8').readlines()
    # nineteen_submit = open('../验证集提交/第十九次（前六模型集成）/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8')
    # twenty_submit = open('../验证集提交/第二十次（合并19和3）/系统之神与我同在_valid_result.txt', 'w+',
    #                        encoding='utf-8')
    # trick_1(third_submit, nineteen_submit, twenty_submit)

    # twenty_submit = open('../验证集提交/第二十次（合并19和3）/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8').readlines()
    # thirteen_submit = open('../验证集提交/第十三次（数据按整个样本进行不切断划分训练，对比第五次）/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8')
    # twenty_one_submit = open('../验证集提交/第二十一次（合并20和13）/系统之神与我同在_valid_result.txt', 'w+',
    #                        encoding='utf-8')
    # trick_1(twenty_submit, thirteen_submit, twenty_one_submit)

    # twenty_submit = open('../验证集提交/第二十次（合并19和3）/系统之神与我同在_valid_result.txt', 'r',
    #                        encoding='utf-8')
    # twenty_three_submit = open('../验证集提交/第23次（去除不合法实体，加上英文标点）/系统之神与我同在_valid_result.txt', 'w+',
    #                          encoding='utf-8')
    # get_rid_of_long_entity(twenty_submit, twenty_three_submit)

    # 提交结果证明 并没有考虑span的评测，所以不需要去重
    # twenty_three_submit = open('../验证集提交/第23次（去除不合法实体，加上英文标点）/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # twenty_four_submit = open('../验证集提交/第24次（实体去重）/系统之神与我同在_valid_result.txt', 'w+',
    #                          encoding='utf-8')
    # duplicate_entity(twenty_three_submit, twenty_four_submit)

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

    # third_submit = open('../验证集提交/第三次/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8').readlines()
    # twenty_five_submit = open('../验证集提交/第25次（前8模型集成）/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8')
    # twenty_six_submit = open('../验证集提交/第26次（合并25和3,去除不合法实体）/系统之神与我同在_valid_result.txt', 'w+',
    #                          encoding='utf-8')
    # trick_1(third_submit, twenty_five_submit, twenty_six_submit)
    # twenty_six_submit = open('../验证集提交/第26次（合并25和3,去除不合法实体）/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # twenty_six_submit_2 = open('../验证集提交/第26次（合并25和3,去除不合法实体）/系统之神与我同在_valid_result_2.txt', 'w+',
    #                          encoding='utf-8')
    # get_rid_of_long_entity(twenty_six_submit, twenty_six_submit_2)
    #
    # twenty_six_submit_2 = open('../验证集提交/第26次（合并25和3,去除不合法实体）/系统之神与我同在_valid_result_2.txt', 'r',
    #                                                     encoding='utf-8')
    # eighteen_submit = open('../验证集提交/第十八次（合并17和13）/系统之神与我同在_valid_result.txt', 'r',
    #                                                encoding='utf-8')
    # twenty_seven_submit = open('../验证集提交/第27次（瞎合并）/系统之神与我同在_valid_result.txt', 'w+',
    #                            encoding='utf-8')
    # trick_1(twenty_six_submit_2, eighteen_submit, twenty_seven_submit)
    # nineteen_submit = open('../验证集提交/第十九次（前六模型集成）/系统之神与我同在_valid_result.txt', 'r',
    #                            encoding='utf-8')
    # twenty_seven_submit = open('../验证集提交/第27次（瞎合并）/系统之神与我同在_valid_result.txt', 'r',
    #                            encoding='utf-8')
    # twenty_seven_submit_2 = open('../验证集提交/第27次（瞎合并）/系统之神与我同在_valid_result_2.txt', 'w+',
    #                            encoding='utf-8')
    # trick_1(nineteen_submit, twenty_seven_submit, twenty_seven_submit_2)

    # third_submit = open('../验证集提交/第三次/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8').readlines()
    # twenty_eight_submit = open('../验证集提交/第28次（addinfo，对比第13次）/系统之神与我同在_valid_result.txt', 'r',
    #                            encoding='utf-8')
    # twenty_nine_submit = open('../验证集提交/第29次（合并28和3）/系统之神与我同在_valid_result.txt', 'w+',
    #                              encoding='utf-8')
    # trick_1(third_submit, twenty_eight_submit, twenty_nine_submit)

    # third_submit = open('../验证集提交/第三次/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8').readlines()
    # thirty_one_submit = open('../验证集提交/第31次（addinfo，集成9模型）/系统之神与我同在_valid_result.txt', 'r',
    #                            encoding='utf-8')
    # thirty_two_submit = open('../验证集提交/第32次（合并31和3）/系统之神与我同在_valid_result.txt', 'w+',
    #                           encoding='utf-8')
    # trick_1(third_submit, thirty_one_submit, thirty_two_submit)
    #
    # thirty_two_submit = open('../验证集提交/第32次（合并31和3）/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # thirty_two_submit_2 = open('../验证集提交/第32次（合并31和3）/系统之神与我同在_valid_result_2.txt', 'w+',
    #                          encoding='utf-8')
    # get_rid_of_long_entity(thirty_two_submit, thirty_two_submit_2)

    # twenty_five_submit = open('../验证集提交/第25次（前8模型集成）/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8')
    # thirty_one_submit = open('../验证集提交/第31次（addinfo，集成9模型）/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # thirty_three_submit = open('../验证集提交/第33次（合并25和31）/系统之神与我同在_valid_result.txt', 'w+',
    #                          encoding='utf-8')
    # trick_1(twenty_five_submit, thirty_one_submit, thirty_three_submit)

    # thirty_four_submit = open('../验证集提交/第34次（add_info_no_O对比第十三次）/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # thirty_four_submit_2 = open('../验证集提交/第34次（add_info_no_O对比第十三次）/系统之神与我同在_valid_result_2.txt', 'w+',
    #                          encoding='utf-8')
    # get_rid_of_long_entity(thirty_four_submit, thirty_four_submit_2)

    # third_submit = open('../验证集提交/第三次/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8').readlines()
    # thirty_four_submit = open('../验证集提交/第34次（add_info_no_O对比第十三次）/系统之神与我同在_valid_result.txt', 'r',
    #                                                      encoding='utf-8')
    # thirty_five_submit = open('../验证集提交/第35次（合并34和3）/系统之神与我同在_valid_result.txt', 'w+',
    #                          encoding='utf-8')
    # trick_1(third_submit, thirty_four_submit, thirty_five_submit)
    # thirty_five_submit = open('../验证集提交/第35次（合并34和3）/系统之神与我同在_valid_result.txt', 'r',
    #                           encoding='utf-8')
    # thirty_five_submit_2 = open('../验证集提交/第35次（合并34和3）/系统之神与我同在_valid_result_2.txt', 'w+',
    #                           encoding='utf-8')
    # get_rid_of_long_entity(thirty_five_submit, thirty_five_submit_2)

    # thirty_one_submit = open('../验证集提交/第31次（addinfo，集成9模型）/系统之神与我同在_valid_result.txt', 'r',
    #                                                     encoding='utf-8')
    # thirty_four_submit = open('../验证集提交/第34次（add_info_no_O对比第十三次）/系统之神与我同在_valid_result.txt', 'r',
    #                           encoding='utf-8')
    # thirty_six_submit = open('../验证集提交/第36次（合并31和34）/系统之神与我同在_valid_result.txt', 'w+',
    #                           encoding='utf-8')
    # trick_1(thirty_one_submit, thirty_four_submit, thirty_six_submit)
    # thirty_six_submit = open('../验证集提交/第36次（合并31和34）/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # thirty_six_submit_2 = open('../验证集提交/第36次（合并31和34）/系统之神与我同在_valid_result_2.txt', 'w+',
    #                             encoding='utf-8')
    # get_rid_of_long_entity(thirty_six_submit, thirty_six_submit_2)

    valid_file_dir = open('../data/dev/ccks_task1_eval_data.txt', 'r', encoding='utf-8')
    submit_r = open('../验证集提交/第34次（add_info_no_O对比第十三次）/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8')
    submit_w = open('../验证集提交/第34次（add_info_no_O对比第十三次）/系统之神与我同在_valid_result_add_text.txt', 'w+', encoding='utf-8')
    add_text2submit(valid_file_dir, submit_r, submit_w)