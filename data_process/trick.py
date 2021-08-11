# -*- coding: utf-8 -*-
# @Time     : 2021/5/25 17:46
# @Author   : 宁星星
# @Email    : shenzimin0@gmail.com
import json
import re
from copy import deepcopy


def trick_1(input_1, input_2, output, except_list=None):
    if not except_list:
        except_list = []
    else:
        print(f'不合并的实体：{except_list}')
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
            if att_ref['type'] in except_list:
                continue
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
    entity_num = 0
    punctuation = """。！？｡＂，"＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"""
    re_punctuation = "[{}]+".format(punctuation)
    for line in inp:
        attributes_list = []
        new_line = dict()
        line = line.strip()
        line = json.loads(line)
        ids = line['text_id']
        attributes = deepcopy(line["attributes"])
        new_line['text_id'] = ids
        new_attributes = deepcopy(line["attributes"])
        for att in attributes:
            att_new = {"entity": att['entity'], "type": att['type']}
            if att_new in attributes_list:
                new_attributes.remove(att)
            else:
                attributes_list.append(att_new)
        attributes = deepcopy(new_attributes)
        for att in attributes:
            if att['type'] != "案发时间":  #  and not att['entity'].endswith('元')
                if re.search(re_punctuation, att['entity']) or len(att['entity']) > 20:
            # if len(att['entity']) > 20:
                    print(att['entity']+'*****'+att['type'])
                    new_attributes.remove(att)
        new_line['attributes'] = new_attributes
        entity_num += len(new_attributes)
        json.dump(new_line, output, ensure_ascii=False)
        output.write('\n')
    print(f'实体数目： {entity_num}个')


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

    # valid_file_dir = open('../data/dev/ccks_task1_eval_data.txt', 'r', encoding='utf-8')
    # submit_r = open('../验证集提交/第34次（add_info_no_O对比第十三次）/系统之神与我同在_valid_result_new_split.txt', 'r', encoding='utf-8')
    # submit_w = open('../验证集提交/第34次（add_info_no_O对比第十三次）/系统之神与我同在_valid_result_new_split_add_text.txt', 'w+', encoding='utf-8')
    # add_text2submit(valid_file_dir, submit_r, submit_w)

    # thirty_nine_submit = open('../验证集提交/39次（complete_label）/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # thirty_nine_submit_2 = open('../验证集提交/39次（complete_label）/系统之神与我同在_valid_result_2.txt', 'w+',
    #                             encoding='utf-8')
    # get_rid_of_long_entity(thirty_nine_submit, thirty_nine_submit_2)

    # thirty_nine_submit = open('../验证集提交/39次（complete_label）/系统之神与我同在_valid_result.txt', 'r',
    #                                                     encoding='utf-8')
    # third_submit = open('../验证集提交/第三次/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8').readlines()
    # forty_submit = open('../验证集提交/40(合并39和3)/系统之神与我同在_valid_result.txt', 'w+',
    #                           encoding='utf-8')
    # trick_1(thirty_nine_submit, third_submit, forty_submit)
    # forty_submit = open('../验证集提交/40(合并39和3)/系统之神与我同在_valid_result.txt', 'r',
    #                     encoding='utf-8')
    # forty_submit_2 = open('../验证集提交/40(合并39和3)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                     encoding='utf-8')
    # get_rid_of_long_entity(forty_submit, forty_submit_2)

    # forty_one_submit = open('../验证集提交/41（集成6模型）/系统之神与我同在_valid_result.txt', 'r',
    #                     encoding='utf-8')
    # forty_on_submit_2 = open('../验证集提交/41（集成6模型）/系统之神与我同在_valid_result_2.txt', 'w+',
    #                     encoding='utf-8')
    # get_rid_of_long_entity(forty_one_submit, forty_on_submit_2)

    # forty_two_submit = open('../验证集提交/42（41采用cut_2）/系统之神与我同在_valid_result.txt', 'r',
    #                         encoding='utf-8')
    # forty_two_submit_2 = open('../验证集提交/42（41采用cut_2）/系统之神与我同在_valid_result_2.txt', 'w+',
    #                          encoding='utf-8')
    # get_rid_of_long_entity(forty_two_submit, forty_two_submit_2)

    # forty_one_submit = open('../验证集提交/41（集成6模型）/系统之神与我同在_valid_result.txt', 'r',
    #                                             encoding='utf-8')
    # nineteen_submit = open('../验证集提交/第十九次（前六模型集成）/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8')
    # forty_three_submit = open('../验证集提交/43(合并41和19)/系统之神与我同在_valid_result.txt', 'w+',
    #                           encoding='utf-8')
    # trick_1(forty_one_submit, nineteen_submit, forty_three_submit)
    # forty_three_submit = open('../验证集提交/43(合并41和19)/系统之神与我同在_valid_result.txt', 'r',
    #                           encoding='utf-8')
    # forty_three_submit_2 = open('../验证集提交/43(合并41和19)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                           encoding='utf-8')
    # get_rid_of_long_entity(forty_three_submit, forty_three_submit_2)

    # forty_one_submit = open('../验证集提交/41（集成6模型）/系统之神与我同在_valid_result.txt', 'r',
    #                           encoding='utf-8')
    # thirty_one_submit = open('../验证集提交/第31次（addinfo，集成9模型）/系统之神与我同在_valid_result.txt', 'r',
    #                                                     encoding='utf-8')
    # forty_four_submit = open('../验证集提交/44(合并41和31)/系统之神与我同在_valid_result.txt', 'w+',
    #                           encoding='utf-8')
    # trick_1(forty_one_submit, thirty_one_submit, forty_four_submit)
    # forty_four_submit = open('../验证集提交/44(合并41和31)/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # forty_four_submit_2 = open('../验证集提交/44(合并41和31)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                             encoding='utf-8')
    # get_rid_of_long_entity(forty_four_submit, forty_four_submit_2)

    # forty_five_submit = open('../验证集提交/45(add_info_with_O_complete_label集成3模型)/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # forty_five_submit_2 = open('../验证集提交/45(add_info_with_O_complete_label集成3模型)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                            encoding='utf-8')
    # get_rid_of_long_entity(forty_five_submit, forty_five_submit_2)

    # forty_six_submit = open('../验证集提交/46(集成41和31的5+9个模型)/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # forty_six_submit_2 = open('../验证集提交/46(集成41和31的5+9个模型)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                            encoding='utf-8')
    # get_rid_of_long_entity(forty_six_submit, forty_six_submit_2)

    # forty_one_submit = open('../验证集提交/41（集成6模型）/系统之神与我同在_valid_result.txt', 'r',
    #                           encoding='utf-8')
    # forty_five_submit = open('../验证集提交/45(add_info_with_O_complete_label集成3模型)/系统之神与我同在_valid_result.txt', 'r',
    #                                                   encoding='utf-8')
    # forty_seven_submit = open('../验证集提交/47（合并45和41）/系统之神与我同在_valid_result.txt', 'w+',
    #                           encoding='utf-8')
    # trick_1(forty_one_submit, forty_five_submit, forty_seven_submit)
    # forty_seven_submit = open('../验证集提交/47（合并45和41）/系统之神与我同在_valid_result.txt', 'r',
    #                           encoding='utf-8')
    # forty_seven_submit_2 = open('../验证集提交/47（合并45和41）/系统之神与我同在_valid_result_2.txt', 'w+',
    #                             encoding='utf-8')
    # get_rid_of_long_entity(forty_seven_submit, forty_seven_submit_2)

    # thirty_one_submit = open('../验证集提交/第31次（addinfo，集成9模型）/系统之神与我同在_valid_result.txt', 'r',
    #                                                                              encoding='utf-8')
    # forty_five_submit = open('../验证集提交/45(add_info_with_O_complete_label集成3模型)/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # forty_eight_submit = open('../验证集提交/48(合并45和31)/系统之神与我同在_valid_result.txt', 'w+',
    #                           encoding='utf-8')
    # trick_1(thirty_one_submit, forty_five_submit, forty_eight_submit)
    # forty_eight_submit = open('../验证集提交/48(合并45和31)/系统之神与我同在_valid_result.txt', 'r',
    #                           encoding='utf-8')
    # forty_eight_submit_2 = open('../验证集提交/48(合并45和31)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                             encoding='utf-8')
    # get_rid_of_long_entity(forty_eight_submit, forty_eight_submit_2)

    # forty_six_submit = open('../验证集提交/46(集成41和31的5+9个模型)/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # forty_five_submit = open('../验证集提交/45(add_info_with_O_complete_label集成3模型)/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # forty_nine_submit = open('../验证集提交/49(合并45和46)/系统之神与我同在_valid_result.txt', 'w+',
    #                           encoding='utf-8')
    # trick_1(forty_six_submit, forty_five_submit, forty_nine_submit)
    # forty_nine_submit = open('../验证集提交/49(合并45和46)/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # forty_nine_submit_2 = open('../验证集提交/49(合并45和46)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                             encoding='utf-8')
    # get_rid_of_long_entity(forty_nine_submit, forty_nine_submit_2)

    # fifty_submit = open('../验证集提交/50（add_info_with_O_complete_label集成7模型）/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # fifty_submit_2 = open('../验证集提交/50（add_info_with_O_complete_label集成7模型）/系统之神与我同在_valid_result_2.txt', 'w+',
    #                             encoding='utf-8')
    # get_rid_of_long_entity(fifty_submit, fifty_submit_2)

    # fifty_submit = open('../验证集提交/50（add_info_with_O_complete_label集成7模型）/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # thirty_one_submit = open('../验证集提交/第31次（addinfo，集成9模型）/系统之神与我同在_valid_result.txt', 'r',
    #                                                                              encoding='utf-8')
    # fifty_one_submit = open('../验证集提交/51(50和31)/系统之神与我同在_valid_result.txt', 'w+',
    #                           encoding='utf-8')
    # trick_1(fifty_submit, thirty_one_submit, fifty_one_submit)
    # fifty_one_submit = open('../验证集提交/51(50和31)/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # fifty_one_submitt_2 = open('../验证集提交/51(50和31)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                             encoding='utf-8')
    # get_rid_of_long_entity(fifty_one_submit, fifty_one_submitt_2)

    # fifty_submit = open('../验证集提交/50（add_info_with_O_complete_label集成7模型）/系统之神与我同在_valid_result.txt', 'r',
    #                     encoding='utf-8')
    # forty_six_submit = open('../验证集提交/46(集成41和31的5+9个模型)/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # fifty_two_submit = open('../验证集提交/52(50和46)/系统之神与我同在_valid_result.txt', 'w+',
    #                         encoding='utf-8')
    # trick_1(fifty_submit, forty_six_submit, fifty_two_submit)
    # fifty_two_submit = open('../验证集提交/52(50和46)/系统之神与我同在_valid_result.txt', 'r',
    #                         encoding='utf-8')
    # fifty_two_submit_2 = open('../验证集提交/52(50和46)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                            encoding='utf-8')
    # get_rid_of_long_entity(fifty_two_submit, fifty_two_submit_2)

    # fifty_three_submit = open('../验证集提交/53（shuffle2021）/系统之神与我同在_valid_result.txt', 'r',
    #                         encoding='utf-8')
    # fifty_three_submit_2 = open('../验证集提交/53（shuffle2021）/系统之神与我同在_valid_result_2.txt', 'w+',
    #                            encoding='utf-8')
    # get_rid_of_long_entity(fifty_three_submit, fifty_three_submit_2)

    # fifty_four_submit = open('../验证集提交/54（shuffle2022）/系统之神与我同在_valid_result.txt', 'r',
    #                           encoding='utf-8')
    # fifty_four_submit_2 = open('../验证集提交/54（shuffle2022）/系统之神与我同在_valid_result_2.txt', 'w+',
    #                             encoding='utf-8')
    # get_rid_of_long_entity(fifty_four_submit, fifty_four_submit_2)

    # fifty_three_submit = open('../验证集提交/53（shuffle2021）/系统之神与我同在_valid_result.txt', 'r',
    #                         encoding='utf-8')
    # fifty_submit = open('../验证集提交/50（add_info_with_O_complete_label集成7模型）/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # fifty_five_submit = open('../验证集提交/55(53和50)/系统之神与我同在_valid_result.txt', 'w+',
    #                         encoding='utf-8')
    # trick_1(fifty_three_submit, fifty_submit, fifty_five_submit)
    # fifty_five_submit = open('../验证集提交/55(53和50)/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # fifty_five_submit_2 = open('../验证集提交/55(53和50)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                            encoding='utf-8')
    # get_rid_of_long_entity(fifty_five_submit, fifty_five_submit_2)

    # fifty_six_submit = open('../验证集提交/56（shuffle集成9模型）/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # fifty_six_submit_2 = open('../验证集提交/56（shuffle集成9模型）/系统之神与我同在_valid_result_2.txt', 'w+',
    #                            encoding='utf-8')
    # get_rid_of_long_entity(fifty_six_submit, fifty_six_submit_2)

    # fifty_eight_submit_tmp = open('../验证集提交/58（56-50投票）/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # thirty_one_submit = open('../验证集提交/第31次（addinfo，集成9模型）/系统之神与我同在_valid_result.txt', 'r',
    #                                                                              encoding='utf-8')
    # fifty_eight_submit = open('../验证集提交/58(56-50投票合并31)/系统之神与我同在_valid_result.txt', 'w+',
    #                         encoding='utf-8')
    # trick_1(fifty_eight_submit_tmp, thirty_one_submit, fifty_eight_submit)
    # fifty_eight_submit = open('../验证集提交/58(56-50投票合并31)/系统之神与我同在_valid_result.txt', 'r',
    #                           encoding='utf-8')
    # fifty_eight_submit_2 = open('../验证集提交/58(56-50投票合并31)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                            encoding='utf-8')
    # get_rid_of_long_entity(fifty_eight_submit, fifty_eight_submit_2)

    # fifty_nine_submit = open('../验证集提交/59(r0.9seed2021)/系统之神与我同在_valid_result.txt', 'r',
    #                           encoding='utf-8')
    # fifty_nine_submit_2 = open('../验证集提交/59(r0.9seed2021)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                            encoding='utf-8')
    # get_rid_of_long_entity(fifty_nine_submit, fifty_nine_submit_2)

    # fifty_eight_submit_tmp = open('../验证集提交/58（56-50投票）/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # twenty_five_submit = open('../验证集提交/第25次（前8模型集成）/系统之神与我同在_valid_result.txt', 'r',
    #                                                                              encoding='utf-8')
    # sixty_submit = open('../验证集提交/60（56-50投票合并25）/系统之神与我同在_valid_result.txt', 'w+',
    #                         encoding='utf-8')
    # trick_1(fifty_eight_submit_tmp, twenty_five_submit, sixty_submit)
    # sixty_submit = open('../验证集提交/60（56-50投票合并25）/系统之神与我同在_valid_result.txt', 'r',
    #                     encoding='utf-8')
    # sixty_submit_2 = open('../验证集提交/60（56-50投票合并25）/系统之神与我同在_valid_result_2.txt', 'w+',
    #                            encoding='utf-8')
    # get_rid_of_long_entity(sixty_submit, sixty_submit_2)

    # sixty_one_submit_tmp = open('../验证集提交/61（56-50-25投票）/系统之神与我同在_valid_result.txt', 'r',
    #                               encoding='utf-8')
    # thirty_one_submit = open('../验证集提交/第31次（addinfo，集成9模型）/系统之神与我同在_valid_result.txt', 'r',
    #                                                                              encoding='utf-8')
    # sixty_one_submit = open('../验证集提交/61(56-50-25投票合并31)/系统之神与我同在_valid_result.txt', 'w+',
    #                     encoding='utf-8')
    # trick_1(thirty_one_submit, sixty_one_submit_tmp, sixty_one_submit)
    # sixty_one_submit = open('../验证集提交/61(56-50-25投票合并31)/系统之神与我同在_valid_result.txt', 'r',
    #                         encoding='utf-8')
    # sixty_one_submit_2 = open('../验证集提交/61(56-50-25投票合并31)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                       encoding='utf-8')
    # get_rid_of_long_entity(sixty_one_submit, sixty_one_submit_2)

    # sixty_two_submit = open('../验证集提交/62(56-50-25投票)/系统之神与我同在_valid_result.txt', 'r',
    #                         encoding='utf-8')
    # sixty_two_submit_2 = open('../验证集提交/62(56-50-25投票)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                       encoding='utf-8')
    # get_rid_of_long_entity(sixty_two_submit, sixty_two_submit_2)

    # sixty_three_submit = open('../验证集提交/63(56-50-31-25投票)/系统之神与我同在_valid_result.txt', 'r',
    #                         encoding='utf-8')
    # sixty_three_submit_2 = open('../验证集提交/63(56-50-31-25投票)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                           encoding='utf-8')
    # get_rid_of_long_entity(sixty_three_submit, sixty_three_submit_2)

    # sixty_three_submit = open('../验证集提交/63(56-50-31-25投票)/系统之神与我同在_valid_result.txt', 'r',
    #                         encoding='utf-8')
    # thirty_one_submit = open('../验证集提交/第31次（addinfo，集成9模型）/系统之神与我同在_valid_result.txt', 'r',
    #                                                                              encoding='utf-8')
    # sixty_four_submit = open('../验证集提交/64（63和31）/系统之神与我同在_valid_result.txt', 'w+',
    #                     encoding='utf-8')
    # trick_1(thirty_one_submit, sixty_three_submit, sixty_four_submit)
    # sixty_four_submit = open('../验证集提交/64（63和31）/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # sixty_four_submit_2 = open('../验证集提交/64（63和31）/系统之神与我同在_valid_result_2.txt', 'w+',
    #                       encoding='utf-8')
    # get_rid_of_long_entity(sixty_four_submit, sixty_four_submit_2)

    # sixty_three_submit = open('../验证集提交/63(56-50-31-25投票)/系统之神与我同在_valid_result.txt', 'r',
    #                           encoding='utf-8')
    # thirty_one_submit = open('../验证集提交/第31次（addinfo，集成9模型）/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # sixty_five_submit = open('../验证集提交/65(63合31（not 受害人身份）)/系统之神与我同在_valid_result.txt', 'w+',
    #                          encoding='utf-8')
    # trick_1(thirty_one_submit, sixty_three_submit, sixty_five_submit, ['受害人身份'])
    # sixty_five_submit = open('../验证集提交/65(63合31（not 受害人身份）)/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # sixty_five_submit_2 = open('../验证集提交/65(63合31（not 受害人身份）)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                            encoding='utf-8')
    # get_rid_of_long_entity(sixty_five_submit, sixty_five_submit_2)

    # sixty_three_submit = open('../验证集提交/63(56-50-31-25投票)/系统之神与我同在_valid_result.txt', 'r',
    #                           encoding='utf-8')
    # thirty_one_submit = open('../验证集提交/第31次（addinfo，集成9模型）/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # sixty_six_submit = open('../验证集提交/66（63合31not受害人身份、涉案平台）/系统之神与我同在_valid_result.txt', 'w+',
    #                          encoding='utf-8')
    # trick_1(thirty_one_submit, sixty_three_submit, sixty_six_submit, ['受害人身份', '涉案平台'])  # ['受害人身份', '涉案平台', '案发城市'] not working
    # sixty_six_submit = open('../验证集提交/66（63合31not受害人身份、涉案平台）/系统之神与我同在_valid_result.txt', 'r',
    #                         encoding='utf-8')
    # sixty_six_submit_2 = open('../验证集提交/66（63合31not受害人身份、涉案平台）/系统之神与我同在_valid_result_2.txt', 'w+',
    #                            encoding='utf-8')
    # get_rid_of_long_entity(sixty_six_submit, sixty_six_submit_2)

    # sixty_three_submit = open('../验证集提交/63(56-50-31-25投票)/系统之神与我同在_valid_result.txt', 'r',
    #                           encoding='utf-8')
    # thirty_one_submit = open('../验证集提交/第31次（addinfo，集成9模型）/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # sixty_seven_submit = open('../验证集提交/67(63合31not受害人身份、涉案平台、案发城市)/系统之神与我同在_valid_result.txt', 'w+',
    #                         encoding='utf-8')
    # trick_1(thirty_one_submit, sixty_three_submit, sixty_seven_submit, ['受害人身份', '涉案平台', '案发城市'])  # ['受害人身份', '涉案平台', '案发城市'] not working
    # sixty_seven_submit = open('../验证集提交/67(63合31not受害人身份、涉案平台、案发城市)/系统之神与我同在_valid_result.txt', 'r',
    #                           encoding='utf-8')
    # sixty_seven_submit_2 = open('../验证集提交/67(63合31not受害人身份、涉案平台、案发城市)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                           encoding='utf-8')
    # get_rid_of_long_entity(sixty_seven_submit, sixty_seven_submit_2)

    # sixty_eight_submit = open('../验证集提交/68（shuffle_r0.9六model）/系统之神与我同在_valid_result.txt', 'r',
    #                           encoding='utf-8')
    # sixty_eight_submit_2 = open('../验证集提交/68（shuffle_r0.9六model）/系统之神与我同在_valid_result_2.txt', 'w+',
    #                           encoding='utf-8')
    # get_rid_of_long_entity(sixty_eight_submit, sixty_eight_submit_2)

    # sixty_nine_submit = open('../验证集提交/69（68-56-50-31-25）/系统之神与我同在_valid_result.txt', 'r',
    #                           encoding='utf-8')
    # sixty_nine_submit_2 = open('../验证集提交/69（68-56-50-31-25）/系统之神与我同在_valid_result_2.txt', 'w+',
    #                             encoding='utf-8')
    # get_rid_of_long_entity(sixty_nine_submit, sixty_nine_submit_2)

    # sixty_nine_submit = open('../验证集提交/69（68-56-50-31-25）/系统之神与我同在_valid_result.txt', 'r',
    #                           encoding='utf-8')
    # thirty_one_submit = open('../验证集提交/第31次（addinfo，集成9模型）/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # seventy_submit = open('../验证集提交/70(69合31)/系统之神与我同在_valid_result.txt', 'w+',
    #                         encoding='utf-8')
    # trick_1(thirty_one_submit, sixty_nine_submit, seventy_submit, ['受害人身份', '涉案平台'])  # ['受害人身份', '涉案平台', '案发城市'] not working
    # seventy_submit = open('../验证集提交/70(69合31)/系统之神与我同在_valid_result.txt', 'r',
    #                       encoding='utf-8')
    # seventy_submit_2 = open('../验证集提交/70(69合31)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                           encoding='utf-8')
    # get_rid_of_long_entity(seventy_submit, seventy_submit_2)

    # seventy_one_submit = open('../验证集提交/71（accumulation_4_step）/系统之神与我同在_valid_result.txt', 'r',
    #                       encoding='utf-8')
    # seventy_one_submit_2 = open('../验证集提交/71（accumulation_4_step）/系统之神与我同在_valid_result_2.txt', 'w+',
    #                         encoding='utf-8')
    # get_rid_of_long_entity(seventy_one_submit, seventy_one_submit_2)

    # seventy_one_submit = open('../验证集提交/71（accumulation_4_step）/系统之神与我同在_valid_result.txt', 'r',
    #                       encoding='utf-8')
    # seventy_submit = open('../验证集提交/70(69合31)/系统之神与我同在_valid_result.txt', 'r',
    #                         encoding='utf-8')
    # seventy_two_submit = open('../验证集提交/72(70合71)/系统之神与我同在_valid_result.txt', 'w+',
    #                         encoding='utf-8')
    # trick_1(seventy_one_submit, seventy_submit, seventy_two_submit, ['受害人身份', '涉案平台'])  # ['受害人身份', '涉案平台', '案发城市'] not working
    # seventy_two_submit = open('../验证集提交/72(70合71)/系统之神与我同在_valid_result.txt', 'r',
    #                           encoding='utf-8')
    # seventy_two_submit_2 = open('../验证集提交/72(70合71)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                           encoding='utf-8')
    # get_rid_of_long_entity(seventy_two_submit, seventy_two_submit_2)

    # twenty_five_submit = open('../验证集提交/第25次（前8模型集成）/系统之神与我同在_valid_result.txt', 'r', encoding='utf-8')
    # seventy_submit = open('../验证集提交/70(69合31)/系统之神与我同在_valid_result.txt', 'r',
    #                       encoding='utf-8')
    # seventy_three_submit = open('../验证集提交/73(70合25)/系统之神与我同在_valid_result.txt', 'w+',
    #                           encoding='utf-8')
    # trick_1(twenty_five_submit, seventy_submit, seventy_three_submit,
    #         ['受害人身份', '涉案平台'])  # ['受害人身份', '涉案平台', '案发城市'] not working
    # seventy_three_submit = open('../验证集提交/73(70合25)/系统之神与我同在_valid_result.txt', 'r',
    #                             encoding='utf-8')
    # seventy_three_submit_2 = open('../验证集提交/73(70合25)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                             encoding='utf-8')
    # get_rid_of_long_entity(seventy_three_submit, seventy_three_submit_2)

    # seventy_four_submit = open('../验证集提交/74(69的选择集成)/系统之神与我同在_valid_result.txt', 'r',
    #                             encoding='utf-8')
    # seventy_four_submit_2 = open('../验证集提交/74(69的选择集成)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                               encoding='utf-8')
    # get_rid_of_long_entity(seventy_four_submit, seventy_four_submit_2)

    # seventy_five_submit = open('../验证集提交/75(maxlen384)/系统之神与我同在_valid_result.txt', 'r',
    #                            encoding='utf-8')
    # seventy_five_submit_2 = open('../验证集提交/75(maxlen384)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                              encoding='utf-8')
    # get_rid_of_long_entity(seventy_five_submit, seventy_five_submit_2)

    # seventy_six_submit = open('../验证集提交/76（maxlen384_testcut350）/系统之神与我同在_valid_result.txt', 'r',
    #                            encoding='utf-8')
    # seventy_six_submit_2 = open('../验证集提交/76（maxlen384_testcut350）/系统之神与我同在_valid_result_2.txt', 'w+',
    #                              encoding='utf-8')
    # get_rid_of_long_entity(seventy_six_submit, seventy_six_submit_2)

    # seventy_seven_submit = open('../验证集提交/77(maxlen384——9model)/系统之神与我同在_valid_result.txt', 'r',
    #                            encoding='utf-8')
    # seventy_seven_submit_2 = open('../验证集提交/77(maxlen384——9model)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                              encoding='utf-8')
    # get_rid_of_long_entity(seventy_seven_submit, seventy_seven_submit_2)


    #seventy_ninetmp_submit = open('../验证集提交/79tmp(77-68-56-50-31-25)/系统之神与我同在_valid_result.txt', 'r',
   #                             encoding='utf-8')
   # thirty_one_submit = open('../验证集提交/第31次（addinfo，集成9模型）/系统之神与我同在_valid_result.txt', 'r',
   #                          encoding='utf-8')
  #  seventy_nine_submit = open('../验证集提交/79(79tmp合31)/系统之神与我同在_valid_result.txt', 'w+',
 #                           encoding='utf-8')
#    trick_1(thirty_one_submit, seventy_ninetmp_submit, seventy_nine_submit, ['受害人身份', '涉案平台'])  # ['受害人身份', '涉案平台', '案发城市'] not working
#
 #   seventy_nine_submit = open('../验证集提交/79(79tmp合31)/系统之神与我同在_valid_result.txt', 'r',
 #                               encoding='utf-8')
#    seventy_nine_submit_2 = open('../验证集提交/79(79tmp合31)/系统之神与我同在_valid_result_2.txt', 'w+',
#                                 encoding='utf-8')
  #  get_rid_of_long_entity(seventy_nine_submit, seventy_nine_submit_2)

    # seventy_nine_tmp_submit = open('../验证集提交/79tmp(77-68-56-50-31-25)/系统之神与我同在_valid_result.txt', 'r',
    #                               encoding='utf-8')
    # seventy_nine_tmp_submit_2 = open('../验证集提交/79tmp(77-68-56-50-31-25)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                                encoding='utf-8')
    # get_rid_of_long_entity(seventy_nine_tmp_submit, seventy_nine_tmp_submit_2)

    # eighty_one_submit = open('../验证集提交/81(R_drop_k_3_10model)/系统之神与我同在_valid_result.txt', 'r',
    #                               encoding='utf-8')
    # eighty_one_submit_2 = open('../验证集提交/81(R_drop_k_3_10model)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                                encoding='utf-8')
    # get_rid_of_long_entity(eighty_one_submit, eighty_one_submit_2)
    #
    # eighty_one_submit = open('../验证集提交/81(R_drop_k_3_10model)/系统之神与我同在_valid_result.txt', 'r',
    #                               encoding='utf-8')
    # thirty_one_submit = open('../验证集提交/第31次（addinfo，集成9模型）/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # eighty_two_submit = open('../验证集提交/82(81合31)/系统之神与我同在_valid_result.txt', 'w+',
    #                         encoding='utf-8')
    # trick_1(thirty_one_submit, eighty_one_submit, eighty_two_submit, ['受害人身份', '涉案平台'])  # ['受害人身份', '涉案平台', '案发城市'] not working
    # eighty_two_submit = open('../验证集提交/82(81合31)/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # eighty_two_submit_2 = open('../验证集提交/82(81合31)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                           encoding='utf-8')
    # get_rid_of_long_entity(eighty_two_submit, eighty_two_submit_2)

    # eighty_four_submit = open('../验证集提交/84(r_drop_k_3_2021)/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # eighty_four_submit_2 = open('../验证集提交/84(r_drop_k_3_2021)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                            encoding='utf-8')
    # get_rid_of_long_entity(eighty_four_submit, eighty_four_submit_2)

    # seventy_ninetmp_submit = open('../验证集提交/79tmp(77-68-56-50-31-25)/系统之神与我同在_valid_result.txt', 'r',
    #                             encoding='utf-8')
    # thirty_one_submit = open('../验证集提交/第31次（addinfo，集成9模型）/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # eighty_five_submit = open('../验证集提交/85(79tmp合31_2)/系统之神与我同在_valid_result.txt', 'w+',
    #                           encoding='utf-8')
    # trick_1(thirty_one_submit, seventy_ninetmp_submit, eighty_five_submit, ['受害人身份', '涉案平台', '案发时间'])  # ['受害人身份', '涉案平台', '案发城市'] not working
    #
    # eighty_five_submit = open('../验证集提交/85(79tmp合31_2)/系统之神与我同在_valid_result.txt', 'r',
    #                           encoding='utf-8')
    # eighty_five_submit_2 = open('../验证集提交/85(79tmp合31_2)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                                 encoding='utf-8')
    # get_rid_of_long_entity(eighty_five_submit, eighty_five_submit_2)

    # seventy_nine_submit = open('../验证集提交/79(79tmp合31)/系统之神与我同在_valid_result.txt', 'r',
    #                               encoding='utf-8')
    # sixty_three_submit = open('../验证集提交/63(56-50-31-25投票)/系统之神与我同在_valid_result.txt', 'r',
    #                           encoding='utf-8')
    # eighty_six_submit = open('../验证集提交/86(79合63)/系统之神与我同在_valid_result.txt', 'w+',
    #                           encoding='utf-8')
    # trick_1(sixty_three_submit, seventy_nine_submit, eighty_six_submit, ['受害人身份', '涉案平台'])  # ['受害人身份', '涉案平台', '案发城市'] not working
    # eighty_six_submit = open('../验证集提交/86(79合63)/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # eighty_six_submit_2 = open('../验证集提交/86(79合63)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                          encoding='utf-8')
    # get_rid_of_long_entity(eighty_six_submit, eighty_six_submit_2)

    # eighty_six_submit = open('../验证集提交/86(79合63)/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # forty_six_submit = open('../验证集提交/46(集成41和31的5+9个模型)/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # eighty_seven_submit = open('../验证集提交/87（86合46）/系统之神与我同在_valid_result.txt', 'w+',
    #                          encoding='utf-8')
    # trick_1(forty_six_submit, eighty_six_submit, eighty_seven_submit,
    #         ['受害人身份', '涉案平台'])  # ['受害人身份', '涉案平台', '案发城市'] not working
    # eighty_seven_submit = open('../验证集提交/87（86合46）/系统之神与我同在_valid_result.txt', 'r',
    #                            encoding='utf-8')
    # eighty_seven_submit_2 = open('../验证集提交/87（86合46）/系统之神与我同在_valid_result_2.txt', 'w+',
    #                            encoding='utf-8')
    # get_rid_of_long_entity(eighty_seven_submit, eighty_seven_submit_2)

    # eighty_six_submit = open('../验证集提交/86(79合63)/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # sixty_nine_submit = open('../验证集提交/69（68-56-50-31-25）/系统之神与我同在_valid_result.txt', 'r',
    #                           encoding='utf-8')
    # eighty_eight_submit = open('../验证集提交/88（86合69）/系统之神与我同在_valid_result.txt', 'w+',
    #                            encoding='utf-8')
    # trick_1(sixty_nine_submit, eighty_six_submit, eighty_eight_submit, ['受害人身份', '涉案平台'])  # ['受害人身份', '涉案平台', '案发城市'] not working
    # eighty_eight_submit = open('../验证集提交/88（86合69）/系统之神与我同在_valid_result.txt', 'r',
    #                            encoding='utf-8')
    # eighty_eight_submit_2 = open('../验证集提交/88（86合69）/系统之神与我同在_valid_result_2.txt', 'w+',
    #                              encoding='utf-8')
    # get_rid_of_long_entity(eighty_eight_submit, eighty_eight_submit_2)

    # eighty_nine_submit = open('../验证集提交/89(macbert集成6model)/系统之神与我同在_valid_result.txt', 'r',
    #                            encoding='utf-8')
    # eighty_nine_submit_2 = open('../验证集提交/89(macbert集成6model)/系统之神与我同在_valid_result_2.txt', 'w+',
    #                              encoding='utf-8')
    # get_rid_of_long_entity(eighty_nine_submit, eighty_nine_submit_2)

    # seventy_ninetmp_submit = open('../验证集提交/79tmp(77-68-56-50-31-25)/系统之神与我同在_valid_result.txt', 'r',
    #                                                           encoding='utf-8')
    # ninety = open('../验证集提交/90（31_new）/系统之神与我同在_valid_result.txt', 'r',
    #                           encoding='utf-8')
    # ninety_onetmp_submit = open('../验证集提交/91（79tmp合31_new合63）/系统之神与我同在_valid_result_tmp.txt', 'w+',
    #                            encoding='utf-8')
    # trick_1(ninety, seventy_ninetmp_submit, ninety_onetmp_submit, ['受害人身份', '涉案平台'])  # ['受害人身份', '涉案平台', '案发城市'] not working
    # ninety_onetmp_submit = open('../验证集提交/91（79tmp合31_new合63）/系统之神与我同在_valid_result_tmp.txt', 'r',
    #                             encoding='utf-8')
    # sixty_three_submit = open('../验证集提交/63(56-50-31-25投票)/系统之神与我同在_valid_result.txt', 'r',
    #                           encoding='utf-8')
    # ninety_one_submit = open('../验证集提交/91（79tmp合31_new合63）/系统之神与我同在_valid_result.txt', 'w+',
    #                            encoding='utf-8')
    # trick_1(sixty_three_submit, ninety_onetmp_submit, ninety_one_submit,
    #         ['受害人身份', '涉案平台'])  # ['受害人身份', '涉案平台', '案发城市'] not working
    # ninety_one_submit = open('../验证集提交/91（79tmp合31_new合63）/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # ninety_one_submit_2 = open('../验证集提交/91（79tmp合31_new合63）/系统之神与我同在_valid_result_2.txt', 'w+',
    #                              encoding='utf-8')
    # get_rid_of_long_entity(ninety_one_submit, ninety_one_submit_2)

    # ninety_two_submit = open('../验证集提交/92（89集成7model）/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # ninety_two_submit_2 = open('../验证集提交/92（89集成7model）/系统之神与我同在_valid_result_2.txt', 'w+',
    #                              encoding='utf-8')
    # get_rid_of_long_entity(ninety_two_submit, ninety_two_submit_2)

    # ninety_two_submit = open('../验证集提交/92（89集成7model）/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # ninety = open('../验证集提交/90（31_new）/系统之神与我同在_valid_result.txt', 'r',
    #                           encoding='utf-8')
    # ninety_three_submit = open('../验证集提交/93（92合31）/系统之神与我同在_valid_result.txt', 'w+',
    #                          encoding='utf-8')
    # trick_1(ninety, ninety_two_submit, ninety_three_submit, ['受害人身份', '涉案平台'])
    # ninety_three_submit = open('../验证集提交/93（92合31）/系统之神与我同在_valid_result.txt', 'r',
    #                            encoding='utf-8')
    # ninety_three_submit_2 = open('../验证集提交/93（92合31）/系统之神与我同在_valid_result_2.txt', 'w+',
    #                              encoding='utf-8')
    # get_rid_of_long_entity(ninety_three_submit, ninety_three_submit_2)

     # ninety_four_submit = open('../验证集提交/94（89集成10model）/系统之神与我同在_valid_result.txt', 'r',
     #                         encoding='utf-8')
     # ninety_four_submit_2 = open('../验证集提交/94（89集成10model）/系统之神与我同在_valid_result_2.txt', 'w+',
     #                            encoding='utf-8')
     # get_rid_of_long_entity(ninety_four_submit, ninety_four_submit_2)

    # ninety_three_submit = open('../验证集提交/93（92合31）/系统之神与我同在_valid_result.txt', 'r',
    #                            encoding='utf-8')
    # sixty_three_submit = open('../验证集提交/63(56-50-31-25投票)/系统之神与我同在_valid_result.txt', 'r',
    #                           encoding='utf-8')
    # ninety_five_submit = open('../验证集提交/95（93合63）/系统之神与我同在_valid_result.txt', 'w+',
    #                          encoding='utf-8')
    # trick_1(sixty_three_submit, ninety_three_submit, ninety_five_submit, ['受害人身份', '涉案平台'])
    # ninety_five_submit = open('../验证集提交/95（93合63）/系统之神与我同在_valid_result.txt', 'r',
    #                           encoding='utf-8')
    # ninety_five_submit_2 = open('../验证集提交/95（93合63）/系统之神与我同在_valid_result_2.txt', 'w+',
    #                              encoding='utf-8')
    # get_rid_of_long_entity(ninety_five_submit, ninety_five_submit_2)


    # ninety_six_submit = open('../验证集提交/96(macbert_alldata)/系统之神与我同在_valid_result_alldata_2021.txt', 'r',
    #                           encoding='utf-8')
    # ninety_six_submit_2 = open('../验证集提交/96(macbert_alldata)/系统之神与我同在_valid_result_alldata_2021_2.txt', 'w+',
    #                              encoding='utf-8')
    # get_rid_of_long_entity(ninety_six_submit, ninety_six_submit_2)

    # ninety_seven_submit_tmp = open('../验证集提交/97(89-56-50-25合31)/系统之神与我同在_valid_result.txt', 'r',
    #                          encoding='utf-8')
    # ninety = open('../验证集提交/90（31_new）/系统之神与我同在_valid_result.txt', 'r',
    #                           encoding='utf-8')
    # ninety_seven_submit = open('../验证集提交/97(89-56-50-25合31)/系统之神与我同在_valid_result_add_31_new.txt', 'w+',
    #                              encoding='utf-8')
    # trick_1(ninety, ninety_seven_submit_tmp, ninety_seven_submit, ['受害人身份', '涉案平台'])
    # ninety_seven_submit = open('../验证集提交/97(89-56-50-25合31)/系统之神与我同在_valid_result_add_31_new.txt', 'r',
    #                            encoding='utf-8')
    # ninety_seven_submit_2 = open('../验证集提交/97(89-56-50-25合31)/系统之神与我同在_valid_result_add_31_new_2.txt', 'w+',
    #                          encoding='utf-8')
    # get_rid_of_long_entity(ninety_seven_submit, ninety_seven_submit_2)

    """**********************************************************"""
    # 测试集
    """**********************************************************"""
    # TODO 第一次提交
    # first_submit = open('../测试集提交/mac_2021/系统之神与我同在_test_result.txt', 'r',
    #                            encoding='utf-8')
    # first_submit_2 = open('../测试集提交/mac_2021/系统之神与我同在_test_result_2.txt', 'w+',
    #                     encoding='utf-8')
    # get_rid_of_long_entity(first_submit, first_submit_2)

    # TODO 第二次提交
    # second_submit = open('../测试集提交/mac_集成5model/系统之神与我同在_test_result.txt', 'r',
    #                     encoding='utf-8')
    # second_submit_2 = open('../测试集提交/mac_集成5model/系统之神与我同在_test_result_2.txt', 'w+',
    #                       encoding='utf-8')
    # get_rid_of_long_entity(second_submit, second_submit_2)

    # TODO 31（小而精）
    # small_31 = open('../测试集提交/31/系统之神与我同在_test_result.txt', 'r',
    #                      encoding='utf-8')
    # small_31_2 = open('../测试集提交/31/系统之神与我同在_test_result_2.txt', 'w+',
    #                        encoding='utf-8')
    # get_rid_of_long_entity(small_31, small_31_2)

    # TODO 第三次提交
    # mac_ten_model = open('../测试集提交/mac集成10model/系统之神与我同在_test_result.txt', 'r',
    #                          encoding='utf-8')
    # small_31 = open('../测试集提交/31/系统之神与我同在_test_result.txt', 'r',
    #                      encoding='utf-8')
    # third_submit = open('../测试集提交/mac集成10model合31/系统之神与我同在_test_result.txt', 'w+',
    #                              encoding='utf-8')
    # trick_1(small_31, mac_ten_model, third_submit, ['受害人身份', '涉案平台'])
    # third_submit = open('../测试集提交/mac集成10model合31/系统之神与我同在_test_result.txt', 'r',
    #                     encoding='utf-8')
    # third_submit_2 = open('../测试集提交/mac集成10model合31/系统之神与我同在_test_result_2.txt', 'w+',
    #                     encoding='utf-8')
    # get_rid_of_long_entity(third_submit, third_submit_2)

    # TODO 第四次提交
    # forth_submit = open('../测试集提交/mac集成10model/系统之神与我同在_test_result.txt', 'r',
    #                     encoding='utf-8')
    # forth_submit_2 = open('../测试集提交/mac集成10model/系统之神与我同在_test_result_2.txt', 'w+',
    #                       encoding='utf-8')
    # get_rid_of_long_entity(forth_submit, forth_submit_2)

    # TODO 第五次提交
    # fifth_submit = open('../测试集提交/56-50-25/系统之神与我同在_test_result.txt', 'r',
    #                     encoding='utf-8')
    # fifth_submit_2 = open('../测试集提交/56-50-25/系统之神与我同在_test_result_2.txt', 'w+',
    #                       encoding='utf-8')
    # get_rid_of_long_entity(fifth_submit, fifth_submit_2)

    # TODO 第六次提交（第二次合第五次）
    second_submit = open('../测试集提交/mac_集成5model/系统之神与我同在_test_result.txt', 'r',
                        encoding='utf-8')
    fifth_submit = open('../测试集提交/56-50-25/系统之神与我同在_test_result.txt', 'r',
                        encoding='utf-8')
    sixth_submit = open('../测试集提交/2合5/系统之神与我同在_test_result.txt', 'w+',
                                 encoding='utf-8')
    trick_1(fifth_submit, second_submit, sixth_submit)  # , ['受害人身份', '涉案平台']
    sixth_submit = open('../测试集提交/2合5/系统之神与我同在_test_result.txt', 'r',
                        encoding='utf-8')
    sixth_submit_2 = open('../测试集提交/2合5/系统之神与我同在_test_result_2.txt', 'w+',
                        encoding='utf-8')
    get_rid_of_long_entity(sixth_submit, sixth_submit_2)