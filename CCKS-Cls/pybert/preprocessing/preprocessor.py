#encoding:utf-8
import re



class ChinesePreProcessor(object):
    def __init__(self,min_len = 2,stopwords_path = None):
        self.min_len = min_len
        self.stopwords_path = stopwords_path
        self.reset()


    def reset(self):
        '''
        加载停用词
        :return:
        '''
        if self.stopwords_path:
            with open(self.stopwords_path,'r') as fr:
                self.stopwords = {}
                for line in fr:
                    word = line.strip(' ').strip('\n')
                    self.stopwords[word] = 1


    def remove_website(self,sentence):
        '''
        处理网址符号
        :param sentence:
        :return:
        '''
        sentence_repl = sentence.replace(r"http\S+", "")
        sentence_repl = sentence_repl.replace(r"https\S+", "")
        sentence_repl = sentence_repl.replace(r"http", "")
        sentence_repl = sentence_repl.replace(r"https", "")
        return sentence_repl



    def remove_breaks(self,sentence):
        # Remove line breaks
        sentence_repl = sentence.replace("\r", "")
        sentence_repl = sentence_repl.replace("\n", "")
        sentence_repl = re.sub(r"\\n\n", ".", sentence_repl)
        return sentence_repl


    def full2half(self,sentence):
        '''
        全角转化为半角
        :param sentence:
        :return:
        '''
        ret_str = ''
        for i in sentence:
            if ord(i) >= 33 + 65248 and ord(i) <= 126 + 65248:
                ret_str += chr(ord(i) - 65248)
            else:
                ret_str += i
        return ret_str

    def remove_stopword(self,sentence):
        '''
        去除停用词
        :param sentence:
        :return:
        '''
        words = sentence.split()
        x = [word for word in words if word not in self.stopwords]
        return " ".join(x)

    # 主函数
    def __call__(self, sentence):
        x = sentence
        # x = self.lower(x)
        x = self.remove_website(x)
        x = self.remove_breaks(x)
        return x
