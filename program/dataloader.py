import codecs
import re
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import names
import time
import matplotlib.pyplot as plt
import pandas as pd

stop_words = ['is', 'be', 'am', 'are', 'was', 'were', 'been',
              'his', 'her', 'their', 'my', 'me', 'your', 'us', 'him'
              'that', 'this', 'those', 'these', 'there', 'here',
              'a', 'an', 'the', 'i', 'he', 'she', 'you', 'they', 'we',
              'in', 'out', 'of', 'to', 'on', 'at', 'off', 'as', 'up', 'down',
              'around', 'for', 'into', 'over', 'by', 'after',
              'before', 'when', 'while', 'so', 'because', 'therefore', 'from', 'although'
              ]
# name_words = [x.lower() for x in names.words('male.txt') + names.words('female.txt')]
name_words = ['ellen', 'tom', 'bob', 'jim', 'jefferson', 'john', 'susan', 'jack', 'jason', 'josh',
              'jerry', 'tina', 'elisa', 'alice', 'harry', 'tony', 'jane', 'jone', 'harry', 'james', 'joe']
lemmatizer = WordNetLemmatizer()


class Preprocess(object):
    def __init__(self):
        self.data = []
        self.lengthDict = defaultdict(int)
        self.difwordsDict = defaultdict(int)
        self.uniDict = defaultdict(int)
        self.blacklst = []  # 暂时不对这些数据进行训练和预测

    # 获取单词词性
    def get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return "a"
        elif tag.startswith('V'):
            return "v"
        elif tag.startswith('N'):
            return "n"
        elif tag.startswith('R'):
            return "r"
        else:
            return "else"

    # 单词词形还原
    def token_lemmatization(self, tokens):
        # print(pos_tag(tokens))
        tags = [self.get_wordnet_pos(x[1]) for x in pos_tag(tokens) if self.get_wordnet_pos(x[1])]
        new_tokens = []
        for i in range(len(tokens)):
            if tags[i] != "else":
                new_tokens.append(lemmatizer.lemmatize(word=tokens[i], pos=tags[i]))
            else:
                new_tokens.append(tokens[i])
        return new_tokens

    # 添加固定规则处理特殊的词汇
    def fixed_rules(self, tokens):
        tokens = re.sub(r"'s", "", tokens)
        tokens = re.sub(r"'ve", "", tokens)
        tokens = re.sub(r"'re", "", tokens)
        tokens = re.sub(r"'m", " am", tokens)
        tokens = re.sub(r"isn't", "is not", tokens)
        tokens = re.sub(r"doesn't", "does not", tokens)
        tokens = re.sub(r"don't", "do not", tokens)
        tokens = re.sub(r"didn't", "did not", tokens)
        tokens = re.sub(r" ran ", " run ", tokens)
        tokens = re.sub(r"new york", "new_york", tokens)
        tokens = re.sub(r"usa", "united_states", tokens)
        return tokens

    def finding_difference(self, sent1, sent2):
        # 分词
        sent1, sent2 = self.fixed_rules(sent1).split(), self.fixed_rules(sent2.strip('\r')).split()
        # 词形还原
        lemmas1, lemmas2 = self.token_lemmatization(sent1), self.token_lemmatization(sent2)
        # 去除停用词
        sent1 = [x for x in lemmas1 if x not in stop_words]
        sent2 = [x for x in lemmas2 if x not in stop_words]

        vocab1, vocab2 = defaultdict(int), defaultdict(int)
        for x in sent1:
            vocab1[x] += 1
        for x in sent2:
            vocab2[x] += 1
        lst1, lst2, same = [], [], []
        for key, value in vocab1.items():
            if value <= vocab2[key] and value != 0:
                same.append(key)
            elif value > vocab2[key]:
                lst1.append(key)
        for key, value in vocab2.items():
            if value <= vocab1[key] and value != 0 and key not in same:
                same.append(key)
            elif value > vocab1[key]:
                lst2.append(key)

        return lst1, lst2, same

    def dataloader(self, dataname):
        dat = []   # 存放数据
        blacklst = []   # 不适于抽取关系的句子
        with codecs.open('../taskA/' + dataname + '.tsv', 'r', encoding='utf_8') as f:
            for line in f:
                newline = {}
                row = line.strip("[\n.,?!']").lower().split('\t')
                ind = int(row[0])  # 数据index
                correct = 1 - int(row[1])
                sent1, sent2 = row[2].replace('.', '').replace(',', '').replace('"', ''), \
                               row[3].replace('.', '').replace(',', '').replace('"', '').strip("\r")

                lst1, lst2, sameword = self.finding_difference(sent1, sent2)

                newline['idx'] = ind
                newline['correct'] = correct
                newline['original'] = sent1.split(), sent2.split()
                newline['diffwords'] = lst1, lst2
                newline['samewords'] = sameword
                dat.append(newline)

                # 两句话没有不一致的部分或相同的词汇位置替换
                if len(sameword) == 0 or len(lst1) == 0 or len(lst2) == 0 \
                        or set(lst1) == set(lst2) or set(sent1.split()) == set(sent2.split()):
                    blacklst.append(ind)
                    # print(ind, lst1, lst2, sameword)

        f.close()
        self.data = dat
        self.blacklst = blacklst

    def datawriter(self, dataname):
        with codecs.open('../dataset/diffwords_' + dataname + '.txt', 'w', encoding='utf_8') as f:
            for i in range(len(self.data)):
                line = self.data[i]
                # if line["idx"] in self.blacklst:
                #     continue
                f.writelines(str(line)+'\n')
        f.close()

    def creating_data_for_bert(self, dataname):
        file = open('../dataset/' + dataname + '_bert.txt', 'w', encoding='utf_8')
        with codecs.open('../taskA/' + dataname + '.tsv', 'r', encoding='utf_8') as f:
            for line in f:
                row = line.strip("[\n.,?!']").lower().split('\t')
                ind = int(row[0])  # 数据index
                correct = 1 - int(row[1])
                sent1, sent2 = row[2].replace('.', '').replace(',', '').replace('"', ''), \
                               row[3].replace('.', '').replace(',', '').replace('"', '').strip('\r')
                # if ind not in self.blacklst:
                newline = {'data_id': ind, 'correct': correct, 'statement1': sent1, 'statement2': sent2}
                file.writelines(str(newline) + '\n')

    def finding_noun_entity(self, filename):
        entitylst = []
        with open("../taskA/" + filename + '.tsv', 'r', encoding='utf_8') as file:
            f = open("../dataset/entity_" + filename + '.txt', 'w', encoding='utf_8')
            for line in file:
                row = line.strip("[\n.,?!']").lower().split('\t')
                sent1, sent2 = row[2].replace('.', '').replace(',', '').replace('"', ''), \
                               row[3].replace('.', '').replace(',', '').replace('"', '')
                for sent in [self.fixed_rules(sent1).split(), self.fixed_rules(sent2).split()]:
                    pos_tags = pos_tag(sent)
                    entity = [lemmatizer.lemmatize(sent[i], "n") for i in range(len(sent))
                              if pos_tags[i][1].startswith('N') and sent[i] not in stop_words]
                    entity = [x for x in list(set(entity)) if x not in name_words]
                    # print(row[0], entity)
                    entitylst.append({"entity": entity})
                    f.writelines(str({"entity": entity}) + '\n')

    def meanlength(self):
        totallength = 0
        for line in self.data:
            totallength += len(line[2]) + len(line[3])
        return totallength / (len(self.data)*2)

    def lengthdict(self):
        Lst = []
        for line in self.data:
            if len(line[2]) + len(line[3]) <= 15:
                self.lengthDict[str(len(line[2]) + len(line[3]))] += 1
            else:
                self.lengthDict['16'] += 1
            if (len(line[2]) + len(line[3])) > 13:
                Lst.append(line[0])
        return Lst


if __name__ == '__main__':

    """
    # 进行对数据集预处理的分析
    dat2 = Preprocess()
    dat2.dataloader2()
    # print(dat2.data[0:10])
    Lst = dat2.lengthdict()
    print(dat2.meanlength())
    print(dat2.lengthDict.items())
    print(Lst)
    # train set 中两句statement不一样的词数分布情况
    lst = sorted(dat2.lengthDict.items(), key=lambda x: int(x[0]), reverse=False)
    x = [ele[0] for ele in lst]
    x[-1] = '>15'
    y = [ele[1] / len(dat2.data) for ele in lst]
    plt.bar(x=x, height=y, color='lightblue', alpha=0.8)
    plt.title('How s1 and s2 differ from each other')
    plt.xlabel("Sum of different words in s1 and s2")
    plt.ylabel("Frequency")
    plt.show()
    plt.savefig("diff_s1s2.png")
    # """

    start = time.time()
    dat1 = Preprocess()
    # """
    dat1.dataloader('train')
    dat1.datawriter('train')
    dat1.creating_data_for_bert('train')    # 写成微调需要的格式
    # print(len(dat1.blacklst))
    # dat1.finding_noun_entity('train')   # 找出句子中出现的名词实体，后续没有再使用这个功能

    dat2 = Preprocess()
    dat2.dataloader('test')
    dat2.datawriter('test')
    dat2.creating_data_for_bert('test')
    # print(len(dat2.blacklst))
    # dat2.finding_noun_entity('test')
    dat3 = Preprocess()
    dat3.dataloader('dev')
    dat3.datawriter('dev')
    dat3.creating_data_for_bert('dev')
    # print(len(dat2.blacklst))
    # dat3.finding_noun_entity('dev')
    
    end = time.time()
    print(end - start)
    # """






