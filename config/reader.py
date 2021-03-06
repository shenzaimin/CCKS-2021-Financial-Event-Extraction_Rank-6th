from tqdm import tqdm
from common import Sentence, Instance
from typing import List
import re
from transformers import BertTokenizer
import json
import copy
import random
import itertools


event_type_map = {
    '兼职刷单类': '0',
    '冒充身份类': '1',
    '用户有泄露信息': '2',
    '其他亲属/其他朋友': '3',
    '广义欺诈类': '4',
    '不发货类': '5',
    '用户泄露信息': '6',
    '伴侣': '7',
    '微信社交负面': '8',
    '设备丢失': '9',
    '违规违禁类': '10',
    '其他': '11',
    '亲友操作': '12',
    '收到商品/服务与约定不符合': '13',
    '直系亲属': '14',
    '欺诈风险': '15',
    '商户风险': '16',
    '胁迫': '17',
    '好友/领导/婚恋交友': '18',
    '无效内容': '19',
    '投资理财/高额返利': '20',
    '信息未泄露': '21',
    '付款后未收到商品|服务': '22',
    '灰产推广（借呗/花呗提额推广/赌博网站等）': '23',
    '色情/赌博/彩票': '24',
    '客服': '25',
    '盗用风险': '26',
    '刷单类': '27'
}


attributes_type_map = {
    '案发城市': 'afcs',
    '受害人': 'shr',
    '受害人身份': 'shrsf',
    '身份证号': 'sfzh',
    '嫌疑人': 'xyr',
    '案发时间': 'afsj',
    '资损金额': 'zsje',
    '涉案平台': 'sapt',
    '银行卡号': 'yhkh',
    '支付渠道': 'zfqd',
    '订单号': 'ddh',
    '手机号': 'sjh',
    '交易号': 'jyh'
}

class Reader:
    def __init__(self, digit2zero: bool=False):
        """
        Read the dataset into Instance
        :param digit2zero: convert the digits into 0, which is a common practice for LSTM-CRF.
        """
        self.digit2zero = digit2zero
        self.vocab = set()

    def find_seg_point(self, line_dict, default_len=500):
        """通过实体的位置，寻找不截断实体的切分位置"""
        seg_points = []
        spans = []
        length = len(line_dict['text'])
        if length <= 500:
            return []
        for mention in line_dict['attributes']:
            spans.append(range(mention['start'], mention['end']+1))
        spans.sort(key=lambda x: x[0])
        spans_nums = list(itertools.chain(*[list(span) for span in spans]))
        seg_point = default_len
        for span in spans:
            if seg_point in span:
                while seg_point in spans_nums:
                    seg_point -= 1
                seg_points.append(seg_point)
                seg_point += default_len
            elif span[0] > seg_point:
                while span[0] > seg_point:
                    seg_points.append(seg_point)
                    seg_point += default_len
                if seg_point < span[-1] + 1:
                    while seg_point in spans_nums:
                        seg_point -= 1
                    seg_points.append(seg_point)
                    seg_point += default_len
            else:
                continue
        for seg_point in range(seg_point, length, default_len):
            seg_points.append(seg_point)
        return seg_points

    def read_txt(self, file_dir: str, number: int = -1, type: str = "all", max_len: int = 500, aug: bool = False) -> List[Instance]:
        """读取原始数据，得到包含text和label序列的inst实例列表"""
        count_0 = 0
        wrong = 0
        entity_num = 0
        entities = []
        origin_entities = []
        insts = []
        inst_combine_list = []
        import os
        # print("Reading file: "+file_dir)
        
        if aug:
            mention_dict = {}
            aug_insts = []
            aug_examples = []
            Examples = []
            for f_temp in os.listdir(file_dir):
                file = os.path.join(file_dir, f_temp)
                in_file = open(file, 'r', encoding = 'utf-8')
                for line in in_file:
                    line = line.strip()
                    Examples.append(json.loads(line))
            for example in Examples:
                content = example['content']
                events = example['events']
                triggers = {}
                reverse_example = copy.deepcopy(example)
                aug_flag = False
                for event in events:
                    sub_flag = False 
                    obj_flag = False
                    for mention in event['mentions']:
                        if not sub_flag and 'sub' in mention['role']:
                            sub_flag = True
                            sub_role = mention['role']
                            sub_word = mention['word']
                        if not obj_flag and 'obj' in mention['role']:
                            obj_flag = True
                            obj_role = mention['role']
                            obj_word = mention['word']
                        if mention['role'] not in mention_dict:
                            mention_dict[mention['role']] = [mention['word']]
                        else:
                            mention_dict[mention['role']].append(mention['word'])
                    if sub_flag and obj_flag and not aug_flag: # swap and replace
                        aug_flag = sub_role.replace('sub', 'aaa') == obj_role.replace('obj', 'aaa')
                if aug_flag:
                    reverse_example['content'] = swap(reverse_example['content'], sub_word, obj_word)
                    for event_id, event in enumerate(reverse_example['events']):
                        for mention_id, mention in enumerate(event['mentions']):
                            if mention['word'] == sub_word:
                                reverse_example['events'][event_id]['mentions'][mention_id]['word'] = obj_word
                            elif mention['word'] == obj_word:
                                reverse_example['events'][event_id]['mentions'][mention_id]['word'] = sub_word
                            temp_word = reverse_example['events'][event_id]['mentions'][mention_id]['word'] 
                            reverse_example['events'][event_id]['mentions'][mention_id]['span'] = get_new_span(content, [sub_word], [obj_word], mention['span'], temp_word)
                    aug_examples.append(reverse_example)
            aug_examples = aug_examples + copy.deepcopy(Examples)
            for example_id, example in enumerate(aug_examples):
                date_flag = True
                number_flag = True
                money_flag = True
                target_flag = True
                proportion_flag = True
                old_words = []
                new_words = []
                temp_content = example['content']
                for event_id, event in enumerate(example['events']):
                    for mention_id, mention in enumerate(event['mentions']):
                        if not date_flag and mention['role'] == 'date':
                            new_date = random.choice(mention_dict['date'])
                            new_words.append(new_date)
                            old_words.append(mention['word'])
                            date_flag = True
                            aug_examples[example_id]['content'] = swap(aug_examples[example_id]['content'], mention['word'], new_date)
                        if not number_flag and mention['role'] == 'number':
                            new_number = random.choice(mention_dict['number'])
                            new_words.append(new_number)
                            old_words.append(mention['word'])
                            number_flag = True
                            aug_examples[example_id]['content'] = swap(aug_examples[example_id]['content'], mention['word'], new_number)
                        if not money_flag and mention['role'] == 'money':
                            new_money = random.choice(mention_dict['money'])
                            new_words.append(new_money)
                            old_words.append(mention['word'])
                            money_flag = True
                            aug_examples[example_id]['content'] = swap(aug_examples[example_id]['content'], mention['word'], new_money)
                        if not target_flag and mention['role'] == 'target-company':
                            new_target = random.choice(mention_dict['sub-org'] + mention_dict['obj-org'])
                            new_words.append(new_target)
                            old_words.append(mention['word'])
                            target_flag = True
                            aug_examples[example_id]['content'] = swap(aug_examples[example_id]['content'], mention['word'], new_target)
                        if not proportion_flag and mention['role'] == 'proportion':
                            new_proportion = random.choice(mention_dict['proportion'] + mention_dict['share-per'])
                            new_words.append(new_proportion)
                            old_words.append(mention['word'])
                            proportion_flag = True
                            aug_examples[example_id]['content'] = swap(aug_examples[example_id]['content'], mention['word'], new_proportion)

                for event_id, event in enumerate(example['events']):
                    for mention_id, mention in enumerate(event['mentions']):
                        if mention['word'] in old_words:
                            idx = old_words.index(mention['word'])
                            replace_word = new_words[idx]
                            aug_examples[example_id]['events'][event_id]['mentions'][mention_id]['word'] = replace_word
                        replace_word = aug_examples[example_id]['events'][event_id]['mentions'][mention_id]['word']
                        aug_examples[example_id]['events'][event_id]['mentions'][mention_id]['span'] = get_new_span(temp_content, old_words, new_words, mention['span'], replace_word)
            aug_examples = check(aug_examples)
            for line in aug_examples:
                words = line['content']
                idx = line['id']
                if self.digit2zero:
                    words = re.sub('\d', '0', words)
                    count_0 += len(re.findall('0', words))
                words = list(words)
                events_dict = dict()
                for k in line['events']:
                    if not k['type'] in events_dict:
                        events_dict[k['type']] = [k]
                    else:
                        events_dict[k['type']].append(k)
                for t, k_list in events_dict.items():
                # for k in line['events']:
                    trigger = []
                    mentions = {}
                    # 不同事件需要单独标注，因为不同事件的实体会重合
                    labels = ['O'] * len(words)
                    evn_type = t
                    for k in k_list:
                        for i in k['mentions']:
                            start_span = i['span'][0]
                            end_span = i['span'][1]
                            role = i['role']
                            mentions[role] = line['content'][start_span:end_span]
                            if role == "trigger":
                                trigger.append(line['content'][start_span:end_span])
                            if end_span - start_span == 1:
                                labels[start_span] = "B-" + event_type_map[evn_type] + attributes_type_map[role]
                                # labels[start_span] = "B-" + attributes_type_map[role]
                            elif end_span - start_span == 2:
                                labels[start_span] = "B-" + event_type_map[evn_type] + attributes_type_map[role]
                                labels[start_span+1] = "E-" + event_type_map[evn_type] + attributes_type_map[role]
                                # labels[start_span] = "B-" + attributes_type_map[role]
                                # labels[start_span+1] = "E-" + attributes_type_map[role]
                            elif end_span - start_span > 2:
                                for i in range(start_span, end_span):
                                    if i == start_span:
                                        labels[i] = "B-" + event_type_map[evn_type] + attributes_type_map[role]
                                        # labels[i] = "B-" + role
                                    elif i == end_span-1:
                                        labels[i] = "E-" + event_type_map[evn_type] + attributes_type_map[role]
                                        # labels[i] = "E-" + role
                                    else:
                                        labels[i] = "I-" + event_type_map[evn_type] + attributes_type_map[role]
                                        # labels[i] = "I-" + role
                            else:
                                print("Wrong span!")
                    aug_inst = Instance(Sentence(words), labels)
                    aug_inst.set_id(idx)
                    aug_inst.type = evn_type
                    aug_inst.trigger = trigger
                    aug_inst.content = line['content']
                    aug_inst.mentions = mentions
                    aug_insts.append(aug_inst)

        for f_temp in os.listdir(file_dir):
            file = os.path.join(file_dir, f_temp)
            print("Reading file: " + file)
            in_file = open(file, 'r', encoding = 'utf-8')
            for line in in_file:
                inst_combine = []  # [[inst,inst],[],...] itertools.chain
                line = line.strip()
                line = json.loads(line)
                seg_points = self.find_seg_point(line, max_len) + [len(line['text'])]
                words_original = line['text']
                idx_original = line['text_id']
                origin_entities.extend(line['attributes'])
                # words = line['text']
                # idx = line['text_id']
                start = 0
                # for i, words in enumerate(self.cut(words_original, 500)):
                for i, seg_point in enumerate(seg_points):
                    idx = f'{idx_original}-{i}'
                    if i == len(seg_points) - 1:
                        words = words_original[start:]
                    else:
                        words = words_original[start:seg_point]
                    line_content_seg = words
                    if self.digit2zero:
                        words = re.sub('\d', '0', words)
                        count_0 += len(re.findall('0', words))

                    words = list(words)
                    events_dict = dict()
                    event_type = '-'.join([line['level1'], line['level2'], line['level3']])
                    events_dict[event_type] = line['attributes']
                    # events_dict[line['level1']] = line['attributes']
                    # events_dict[line['level2']] = line['attributes']
                    # events_dict[line['level3']] = line['attributes']
                    # # for k in line['attributes']:
                    #     if not k['type'] in events_dict:
                    #         events_dict[k['type']] = [k]
                    #     else:
                    #         events_dict[k['type']].append(k)
                    for t, k in events_dict.items():
                    # for k in line['events']:
                        trigger = []
                        mentions = {}
                        # 不同事件需要单独标注，因为不同事件的实体会重合
                        labels = ['O'] * len(words)
                        evn_type = t
                        # for k in k_list:
                        for i in k:
                            if start <= i['start'] < seg_point:
                                entity_num += 1
                                start_span = i['start'] - start
                                end_span = i['end'] + 1 - start
                                role = i['type']

                                entities.append(i)
                                mentions[role] = mentions.get(role, []) + [line['text'][start_span+start:end_span+start]]
                                # check有多少标错的实体
                                if i['entity'] != line['text'][start_span+start:end_span+start]:
                                    start_span = i['start'] - start
                                    end_span = i['end'] + 1 - start
                                    print(
                                        f'{idx_original}-[WRONG]-{line["text"][start_span + start:end_span + start]}-[right]-{i["entity"]}-{start_span + start}')
                                    wrong += 1
                                if self.digit2zero:
                                    if role in ['身份证号', '银行卡号', '订单号', '手机号', '交易号']:
                                        print(i)
                                        continue
                                if end_span - start_span == 1:
                                    # labels[start_span] = "B-" + event_type_map[evn_type] + attributes_type_map[role]
                                    labels[start_span] = "B-" + attributes_type_map[role]
                                elif end_span - start_span == 2:
                                    # labels[start_span] = "B-" + event_type_map[evn_type] + attributes_type_map[role]
                                    # labels[start_span+1] = "E-" + event_type_map[evn_type] + attributes_type_map[role]
                                    labels[start_span] = "B-" + attributes_type_map[role]
                                    labels[start_span+1] = "E-" + attributes_type_map[role]
                                elif end_span - start_span > 2:
                                    for ids in range(start_span, end_span):
                                        if ids == start_span:
                                            # labels[ids] = "B-" + event_type_map[evn_type] + attributes_type_map[role]
                                            labels[ids] = "B-" + attributes_type_map[role]
                                        elif ids == end_span-1:
                                            # labels[ids] = "E-" + event_type_map[evn_type] + attributes_type_map[role]
                                            labels[ids] = "E-" + attributes_type_map[role]
                                        else:
                                            # labels[ids] = "I-" + event_type_map[evn_type] + attributes_type_map[role]
                                            labels[ids] = "I-" + attributes_type_map[role]
                                else:
                                    print("Wrong span!")
                        inst = Instance(Sentence(words), labels)
                        inst.set_id(idx)
                        inst.type = evn_type
                        inst.trigger = trigger
                        inst.content = line_content_seg
                        inst.level1 = line['level1']
                        inst.level2 = line['level2']
                        inst.level3 = line['level3']
                        inst.level_total = f'{inst.level1}-{inst.level2}-{inst.level3}'
                        inst.mentions = mentions
                        insts.append(inst)
                        inst_combine.append(inst)
                        if len(insts) == number:
                            break
                    start = seg_point
                inst_combine_list.append(inst_combine)
        print("numbers being replaced by zero:", count_0)
        print("number of sentences: {}".format(len(insts)))
        assert len(origin_entities) == len(entities)
        print("number of entities: {}".format(len(entities)))
        print(f"{wrong} entities fucking being mismatched!")
        if type == "all":
            return inst_combine_list
        else:
            if aug:
                return [inst for inst in insts if type in inst.type], [inst for inst in aug_insts if type in inst.type]
            else:
                return [inst for inst in insts if type in inst.type]

    def cut(self, text, seg_len):
        str_list = [text[i:i + seg_len] for i in range(0, len(text), seg_len)]
        return str_list

    def cut_2(self, text, seg_len):
        """
        用标点分割
        """
        punc_pattern = r'[,; ]'
        # punc_list = re.findall(punc_pattern, text)
        sub_text_list = [t for t in re.split(punc_pattern, text) if len(t)]
        str_list = []
        sub_str = ""
        for i, sub_text in enumerate(sub_text_list):
            if i == 0:
                sub_str = sub_text
                assert len(sub_str) <= seg_len
                continue
            tmp_str = ','.join([sub_str, sub_text])
            if len(tmp_str) < seg_len:
                sub_str = tmp_str
                if i != len(sub_text_list) - 1:
                    continue
            str_list.append(sub_str+',')
            sub_str = sub_text
            if len(sub_str) > seg_len:
                sub_cut_list = self.cut(sub_str, seg_len)
                str_list.extend(sub_cut_list[:-1])
                sub_str = sub_cut_list[-1]
            # assert len(sub_str) <= seg_len
        return str_list

    def read_test_txt(self, file_dir: str, number: int = -1) -> List[Instance]:
        count_0 = 0
        insts = []
        import os
        # print("Reading file: "+file_dir)
        for f_temp in os.listdir(file_dir):
            file = os.path.join(file_dir, f_temp)
            print("Reading file: " + file)
            in_file = open(file, 'r', encoding='utf-8')
            for line in in_file:
                line = line.strip()
                line = json.loads(line)
                words_original = line['text']
                idx_original = line['text_id']
                for i, words in enumerate(self.cut(words_original, 500)):
                    idx = f'{idx_original}-{i}'
                    seg_content = words
                    if self.digit2zero:
                        words = re.sub('\d', '0', words)
                        count_0 += len(re.findall('0', words))
                    words = list(words)
                    labels = ['O'] * len(words)
                    inst = Instance(Sentence(words), labels)
                    inst.content = line['text']
                    inst.seg_content = seg_content
                    inst.level1 = line['level1']
                    inst.level2 = line['level2']
                    inst.level3 = line['level3']
                    inst.level_total = f'{inst.level1}-{inst.level2}-{inst.level3}'
                    inst.set_test_id(idx)
                    insts.append(inst)
                    if len(insts) == number:
                        break
        print("numbers being replaced by zero:", count_0)
        print("number of sentences: {}".format(len(insts)))
        return insts

    def get_origin_query(self, file: str, number: int = -1):
        print("Reading file: " + file)
        with open(file, 'r') as f:
            f_dict = json.loads(f.read())
        query_list = f_dict.values()
        return query_list

def fix_trigger():
    # 修改valid_result
    id2content = dict()
    for dev in tqdm(devs):
        id2content[dev.id] = dev.content
    import json
    sub_data = open('./result_tmp.json', 'r', encoding='utf-8')
    fix_data = open('./result.json', 'w+', encoding='utf-8')
    cnt_zb = 0
    cnt_sg = 0
    cnt_qsht = 0
    cnt_db = 0
    cnt_pj = 0
    for i, line in tqdm(enumerate(sub_data.readlines())):
        line = line.strip()
        eval_dict = json.loads(line)
        # if i < 8132:
        # if i < 8267:
        for evn in eval_dict['events']:
            if evn['type'] == "中标":
                trigger_flag = False
                for mention in evn['mentions']:
                    if "trigger" in mention.values():
                        trigger_flag = True
                if not trigger_flag:
                    context = id2content[eval_dict['id']]
                    for candidate in ['中标']:
                        span_start = context.find(candidate)
                        if span_start != -1:
                            span_end = span_start + len(candidate)
                            evn['mentions'].append(
                                {"word": candidate, "span": [span_start, span_end], "role": "trigger"})
                            cnt_zb += 1
                            break
            elif evn['type'] == "收购":
                trigger_flag = False
                for mention in evn['mentions']:
                    if "trigger" in mention.values():
                        trigger_flag = True
                if not trigger_flag:
                    context = id2content[eval_dict['id']]
                    for candidate in ['收购', '间接控制', '控制权', '合并']:
                        span_start = context.find(candidate)
                        if span_start != -1:
                            span_end = span_start + len(candidate)
                            evn['mentions'].append(
                                {"word": candidate, "span": [span_start, span_end], "role": "trigger"})
                            cnt_sg += 1
                            break
            elif evn['type'] == "担保":
                trigger_flag = False
                for mention in evn['mentions']:
                    if "trigger" in mention.values():
                        trigger_flag = True
                if not trigger_flag:
                    context = id2content[eval_dict['id']]
                    for candidate in ['担保']:
                        span_start = context.find(candidate)
                        if span_start != -1:
                            span_end = span_start + len(candidate)
                            evn['mentions'].append(
                                {"word": candidate, "span": [span_start, span_end], "role": "trigger"})
                            cnt_db += 1
                            break
            elif evn['type'] == "签署合同":
                trigger_flag = False
                for mention in evn['mentions']:
                    if "trigger" in mention.values():
                        trigger_flag = True
                if not trigger_flag:
                    context = id2content[eval_dict['id']]
                    for candidate in ['签约', '签署合同', '签订', '订立', '合作']:
                        span_start = context.find(candidate)
                        if span_start != -1:
                            span_end = span_start + len(candidate)
                            evn['mentions'].append(
                                {"word": candidate, "span": [span_start, span_end], "role": "trigger"})
                            cnt_qsht += 1
                            break
            elif evn['type'] == "判决":
                trigger_flag = False
                for mention in evn['mentions']:
                    if "trigger" in mention.values():
                        trigger_flag = True
                if not trigger_flag:
                    context = id2content[eval_dict['id']]
                    for candidate in ['判决', '裁决', '审理', '宣判', '判令', '裁定', '判处', '仲裁']:
                        span_start = context.find(candidate)
                        if span_start != -1:
                            span_end = span_start + len(candidate)
                            evn['mentions'].append(
                                {"word": candidate, "span": [span_start, span_end], "role": "trigger"})
                            cnt_qsht += 1
                            break
            else:
                continue
        json.dump(eval_dict, fix_data, ensure_ascii=False)
        fix_data.write("\n")
    print("db: ", cnt_db)
    print("zb: ", cnt_zb)
    print("qsht: ", cnt_qsht)
    print("sg: ", cnt_sg)
    print("pj: ", cnt_pj)

def extract_by_reg():
    pattern_number = r'质押\D*(\d+\.?\d+[万亿])余?股'
    # pattern_number_2 = r'质押\D*(\d+,?\d+,?\d+\.?\d*)余?股'
    pattern_number_2 = r'质押\D*((?:\d+,){0,3}\d+\.?\d*)余?股'
    pattern_number_3 = r'(\d+\.?\d+[万亿])余?股\D*进行.*质押'
    pattern_number_4 = r'(\d+\.?\d+[万亿])余?股\D*质押(?:于|给)'
    pattern_number_5 = r'((?:\d+,){0,3}\d+\.?\d*)余?股\D*进行.*质押'
    pattern_number_6 = r'((?:\d+,){0,3}\d+\.?\d*)余?股\D*质押(于|给)'
    pattern_number_7 = r'(\d+\.?\d+[万亿])余?股\D*(?:(?:被)|(?:处于))质押'
    pattern_number_8 = r'((?:\d+,){0,3}\d+\.?\d*)余?股\D*(?:(?:被)|(?:处于))质押'
    pattern_list = [pattern_number, pattern_number_2, pattern_number_3,
                    pattern_number_4, pattern_number_5, pattern_number_6,
                    pattern_number_7, pattern_number_8]
    p = 0
    t = 0
    err_list = []
    for content, number in [(t.content, t.mentions['number']) for t in trains if
                            t.type == '质押' and 'number' in t.mentions]:
        t += 1
        for pattern in pattern_list:
            if re.search(pattern, content) and re.search(pattern, content).group(1) == number:
                p += 1
                break
        else:
            err_list.append((content, number))
    print(p / t)

def fix_NUM():
    pattern_number = r'质押\D*(\d+\.?\d+[万亿])余?股'
    # pattern_number_2 = r'质押\D*(\d+,?\d+,?\d+\.?\d*)余?股'
    pattern_number_2 = r'质押\D*((?:\d+,){0,3}\d+\.?\d*)余?股'
    pattern_number_3 = r'(\d+\.?\d+[万亿])余?股\D*进行.*质押'
    pattern_number_4 = r'(\d+\.?\d+[万亿])余?股\D*质押(?:于|给)'
    pattern_number_5 = r'((?:\d+,){0,3}\d+\.?\d*)余?股\D*进行.*质押'
    pattern_number_6 = r'((?:\d+,){0,3}\d+\.?\d*)余?股\D*质押(于|给)'
    pattern_number_7 = r'(\d+\.?\d+[万亿])余?股\D*(?:(?:被)|(?:处于))质押'
    pattern_number_8 = r'((?:\d+,){0,3}\d+\.?\d*)余?股\D*(?:(?:被)|(?:处于))质押'
    pattern_list = [pattern_number, pattern_number_2, pattern_number_3,
                    pattern_number_4, pattern_number_5, pattern_number_6,
                    pattern_number_7, pattern_number_8]
    # 修改valid_result
    id2content = dict()
    for dev in tqdm(devs):
        id2content[dev.id] = dev.content
    import json
    sub_data = open('../valid_result_15_fix.json', 'r', encoding='utf-8')
    fix_data = open('../valid_result_15_fix_num.json', 'w+', encoding='utf-8')
    cnt_add = 0
    cnt_fix = 0
    for i, line in tqdm(enumerate(sub_data.readlines())):
        line = line.strip()
        eval_dict = json.loads(line)
        # if i < 8132:
        # if i < 8267:
        for evn in eval_dict['events']:
            if evn['type'] == "质押":
                number_flag = False
                for mention in evn['mentions']:
                    if "number" in mention.values():
                        number_flag = True
                        mention_num = mention
                context = id2content[eval_dict['id']]
                for pattern in pattern_list:
                    if re.search(pattern, context):
                        number = re.search(pattern, context).group(1)
                        span_start = context.find(number)
                        span_end = span_start + len(number)
                        if number_flag:
                            if mention_num != {"word": number, "span": [span_start, span_end], "role": "number"}:
                                evn['mentions'].remove(mention_num)
                                evn['mentions'].append({"word": number, "span": [span_start, span_end], "role": "number"})
                                cnt_fix+=1
                                print(eval_dict['id'])
                        else:
                            evn['mentions'].append(
                                {"word": number, "span": [span_start, span_end], "role": "number"})
                            cnt_add += 1
                        break
            else:
                continue
        json.dump(eval_dict, fix_data, ensure_ascii=False)
        fix_data.write("\n")
    print("add: ", cnt_add)
    print("fix: ", cnt_fix)

def error_output():
    error_data = open('../data/error_info.json', 'w+', encoding='utf8')
    type_map = {
        'zy': '质押',
        'gfgqzr': '股份股权转让',
        'qs': '起诉',
        'tz': '投资',
        'jc': '减持',
        'sg': '收购',
        'pj': '判决'
    }
    for suffix in ['zy', 'gfgqzr', 'qs', 'tz', 'jc', 'sg', 'pj']:
        type = type_map[suffix]
        pres = reader.read_txt(file_dir_err, -1, type)
        for pre in pres:
            err_dict = dict()
            err_dict['type'] = type
            ids = pre.id
            err_dict['id'] = ids
            mentions_pre = pre.mentions
            mentions_true = [t.mentions for t in trains if t.id==ids and t.type==type][0]
            for k, v in mentions_true.items():
                if k in mentions_pre.keys() and v != mentions_pre[k]:
                    err_dict[k] = (v, mentions_pre[k])
                elif k not in mentions_pre.keys():
                    err_dict[k] = (v, '')
            err_dict['content'] = pre.content
            import json
            if len(err_dict.keys()) == 3:
                continue
            json.dump(err_dict, error_data, ensure_ascii=False)
            error_data.write('\n')

def error_output_csv():
    import pandas as pd
    import numpy as np
    error_list = []
    type_map = {
        'zy': '质押',
        'gfgqzr': '股份股权转让',
        'qs': '起诉',
        'tz': '投资',
        'jc': '减持',
        'sg': '收购',
        'pj': '判决'
    }
    for suffix in ['zy', 'gfgqzr', 'qs', 'tz', 'jc', 'sg', 'pj']:
        type = type_map[suffix]
        pres = reader.read_txt(file_dir_err, -1, type)
        for pre in pres:
            ids = pre.id
            mentions_pre = pre.mentions
            mentions_true = [t.mentions for t in trains if t.id == ids and t.type == type][0]
            for k, v in mentions_true.items():
                if k in mentions_pre.keys() and v != mentions_pre[k]:
                    error_list.append([ids, type, k, v, mentions_pre[k], pre.content])
                elif k not in mentions_pre.keys():
                    error_list.append([ids, type, k, v, '', pre.content])
    error_array = np.asarray(error_list)
    df = pd.DataFrame(error_array, index=None)
    df.columns = ['id', 'type', 'role', 'answer', 'predict', 'content']
    df.to_csv('../data/error_info.csv', index=False)

def error_output_csv_2(file_pre, file_true):
    import pandas as pd
    import numpy as np
    error_list = []
    content_dict = dict()
    pre_dict = dict()
    true_dict = dict()
    for line in open(file_pre,'r'):
        line = line.strip()
        line = json.loads(line)
        ids = line['id']
        pre_dict[ids] = line['events']
        content_dict[ids] = line['contents']
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
            for evn_true in true_events:
                if evn_true['type'] == type:
                    mentions_true = dict()
                    for m in evn_true['mentions']:
                        mentions_true[m['role']] = m['word']
                    for k, v in mentions_true.items():
                        if k in mentions_pre.keys() and v != mentions_pre[k]:
                            error_list.append([ids, type, k, v, mentions_pre[k], content])
                        elif k not in mentions_pre.keys():
                            error_list.append([ids, type, k, v, '', content])
    error_array = np.asarray(error_list)
    df = pd.DataFrame(error_array, index=None)
    df.columns = ['id', 'type', 'role', 'answer', 'predict', 'content']
    df.to_csv('../data/error_info.csv', index=False)

def fix_classify():
    print('fixing classify...')
    classify_dict = {
        'base': ['质押', '股份股权转让', '起诉', '投资', '减持'],
        'trans': ['收购', '判决']
    }
    import json
    dev_base = open('../data/dev/dev_base.json', 'r', encoding='utf8')
    dev_trans = open('../data/dev/trans_dev.json', 'r', encoding='utf8')
    base_ids = []
    trans_ids = []
    for line in tqdm(dev_base.readlines()):
        line = line.strip()
        base_dict = json.loads(line)
        base_ids.append(base_dict['id'])
    for line in tqdm(dev_trans.readlines()):
        line = line.strip()
        trans_dict = json.loads(line)
        trans_ids.append(trans_dict['id'])
    sub_data = open('../valid_result_16_fix.json', 'r', encoding='utf-8')
    fix_data = open('../valid_result_16_fix_classify.json', 'w+', encoding='utf-8')
    fix_num = 0
    fix_num_base = 0
    fix_num_trans = 0
    for i, line in tqdm(enumerate(sub_data.readlines())):
        line = line.strip()
        eval_dict = json.loads(line)
        ids = eval_dict['id']
        remove_evns = []
        if ids in base_ids:
            type_evn = 'base'
        else:
            type_evn = 'trans'
        for evn in eval_dict['events']:
            if evn['type'] not in classify_dict[type_evn]:
                remove_evns.append(evn)
                if type_evn == 'base':
                    fix_num_base += 1
                else:
                    fix_num_trans += 1
        for e in remove_evns:
            fix_num += 1
            eval_dict['events'].remove(e)
        json.dump(eval_dict, fix_data, ensure_ascii=False)
        fix_data.write("\n")
    print('total_fix: '+str(fix_num))
    print('base_fix: '+str(fix_num_base))
    print('trans_fix: '+str(fix_num_trans))

def add_trigger():
    sub_data = open('../valid_result_16_fix.json', 'r', encoding='utf-8')
    fix_data = open('../valid_result_16_fix_add_trigger.json', 'w+', encoding='utf-8')

    id_list = []
    add_cnt = 0
    for i, line in tqdm(enumerate(sub_data.readlines())):
        line = line.strip()
        eval_dict = json.loads(line)
        ids = eval_dict['id']
        sub_org_cnt = 0
        trigger_cnt = 0
        for evn in eval_dict['events']:
            if evn['type'] == '股份股权转让':
                for mention in evn['mentions']:
                    if mention['role'] == 'sub-org':
                        sub_org_cnt += 1
                    if mention['role'] == 'trigger':
                        mention_trigger = mention
                        trigger_cnt += 1
                if trigger_cnt != 0:
                    for i in range(sub_org_cnt - trigger_cnt):
                        add_cnt += 1
                        evn['mentions'].append(mention_trigger)
        json.dump(eval_dict, fix_data, ensure_ascii=False)
        fix_data.write("\n")
    print(add_cnt)

# def fix_gfgqzr_tz_pairs():
#     print('fixing pairs...')
#     fix_map = {
#         'target-company': 'obj',
#         'trigger': 'trigger',
#         'money': 'money',
#         'date': 'date',
#         'obj-org': 'sub'
#     }
#     import json
#     sub_data = open('../valid_result_16_fix.json', 'r', encoding='utf-8')
#     fix_data = open('../valid_result_16_fix_gfgqzr_tz_pairs.json', 'w+', encoding='utf-8')
#     for i, line in tqdm(enumerate(sub_data.readlines())):
#         line = line.strip()
#         eval_dict = json.loads(line)
#         origin_evn = dict()
#         fix_evn = dict()
#         for evn in eval_dict['events']:
#             if evn['type']=='股份股权转让' or evn['type']=='投资':
#                 origin_evn[evn['type']] = evn
#         if len(origin_evn)==2:
#             for mention in origin_evn['股份股权转让']['mentions']:
#                 if mention['role'] in fix_map.keys():

def get_dev_ids(file_dir):
    import pandas as pd
    type_map = {
        'zy': '质押',
        'gfgqzr': '股份股权转让',
        'qs': '起诉',
        'tz': '投资',
        'jc': '减持',
        'sg': '收购',
        'pj': '判决'
    }
    dev_ids = []
    for suffix in ['zy', 'gfgqzr', 'qs', 'tz', 'jc', 'sg', 'pj']:
        # read trains
        trains_add_devs = reader.read_txt(file_dir, -1, type_map[suffix])
        trains = trains_add_devs[:int(0.8 * len(trains_add_devs))]
        devs = trains_add_devs[int(0.8 * len(trains_add_devs)):]
        dev_ids.extend([d.id for d in devs])
    dev_ids = list(set(dev_ids))
    df = pd.DataFrame(dev_ids,index=None,columns=['dev_id'])
    df.to_csv('./dev_ids.csv',)
    return df

def add_content2result(devs):
    id2content = dict()
    for d in devs:
        id2content[d.id] = d.content
    sub_data = open('./valid_result.json', 'r', encoding='utf-8')
    fix_data = open('./valid_result_add_content.json', 'w+', encoding='utf-8')
    for i, line in tqdm(enumerate(sub_data.readlines())):
        line = line.strip()
        eval_dict = json.loads(line)
        ids = eval_dict['id']
        eval_dict['contents'] = id2content[ids]
        json.dump(eval_dict, fix_data, ensure_ascii=False)
        fix_data.write("\n")


def check(examples):
    ID_list = []
    good_examples = []
    cnt = 0
    for idx, example in enumerate(examples):
        if example['id'] not in ID_list:
            ID_list.append(example['id'])
        else:
            continue
        flag = True
        content = example['content']
        for event in example['events']:
            for mention in event['mentions']:
                if content[mention['span'][0]:mention['span'][1]] != mention['word']:
                    flag = False
                    #print(mention['span'])
                    #print(mention['word'])
                    #print(content[mention['span'][0]:mention['span'][1]])
        if flag:
            good_examples.append(example)
        else:
            #print(example, idx)
            cnt += 1
    print("Total %d error augmented examples !" %(cnt))
    return good_examples

def swap(sentence, swap1, swap2):
    if swap2 == swap1:
        return sentence
    temp = '````||||'
    sentence = sentence.replace(swap1, temp)
    sentence = sentence.replace(swap2, swap1)
    sentence = sentence.replace(temp, swap2)
    return sentence

def get_new_span(sentence, swaps1, swaps2, old_span, temp_word):
    if old_span[0] == 0:
        return [0, len(temp_word)]
    old_sentence = sentence[:old_span[0]]
    for i in range(len(swaps1)):
        old_sentence = swap(old_sentence, swaps1[i], swaps2[i])
    new_span = [len(old_sentence), len(old_sentence) + len(temp_word)]
    return new_span


if __name__ == "__main__":
    reader = Reader(True)
    file_dir = "../data/train"
    file_dir_test = "../data/dev"
    file_dir_err = '../data/error'
    trains = reader.read_txt(file_dir, -1,"")
    devs = reader.read_test_txt(file_dir_test, -1)
    # get_dev_ids(file_dir)
    # add_content2result(devs)
    # error_output()
    # error_output_csv()
    fix_trigger()
    # extract_by_reg()
    # fix_NUM()
    # fix_classify()
