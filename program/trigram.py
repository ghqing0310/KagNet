import codecs
import math
import time
from collections import defaultdict
import numpy as np


class NgramModel(object):
    def __init__(self):
        self.vocab = set()  # 建立词汇表
        self.vocabnum = 0   # 词汇总数
        self.ntrain = 0     # 训练集样本数
        # 建立n-gram字典
        self.uniDict = defaultdict(int)
        self.biDict = defaultdict(int)
        self.triDict = defaultdict(int)
        self.bi_preDict = defaultdict(int)
        self.tri_preDict = defaultdict(int)
        self.bicount = 0

        self.wrong = []     # 错分类样本idx
        self.senttypeDict = defaultdict(int)

    def dataloader(self):
        traindat = []
        with codecs.open('../taskA/train.tsv', 'r', encoding='utf_8') as f:
            for line in f:
                row = line.strip('[\n.,?!]').lower().split('\t')
                ind = int(row[0])   # 数据index
                correct = 1 - int(row[1])
                corrsent = row[2+correct].split()
                for i in range(len(corrsent)):
                    self.uniDict[corrsent[i]] += 1
                    self.vocab.add(corrsent[i])
                    if i >= 1:
                        self.biDict[corrsent[i] + '|' + corrsent[i-1]] += 1
                        self.bi_preDict[corrsent[i-1]] += 1
                    if i >= 2:
                        self.triDict[corrsent[i] + '|' + corrsent[i-2] + ',' + corrsent[i-1]] += 1
                        self.tri_preDict[corrsent[i-2] + ',' + corrsent[i-1]] += 1

                sents = (correct, row[2], row[3])
                traindat.append([ind, sents])

            self.ntrain = len(traindat)
            self.vocabnum = len(self.vocab) + 1
            self.bicount = len(self.biDict.keys())
        f.close()   # 7007条数据

    def NgramVal(self, sent, alpha, backoff):
        sentProb = 0
        for i in range(len(sent)):
            if i == 0:
                sentProb += math.log((self.uniDict[sent[i]]+alpha) / (sum(self.uniDict.values())+alpha*self.vocabnum))
            elif i == 1:
                sentProb += 0.2 * math.log((self.uniDict[sent[i]]+alpha) / (sum(self.uniDict.values())+alpha*self.vocabnum)) + \
                    0.8 * math.log((self.biDict[sent[i] + '|' + sent[i-1]]+alpha) / (self.bi_preDict[sent[i-1]]+alpha * self.vocabnum))
            else:
                sentProb += backoff[0] * math.log((self.uniDict[sent[i]]+alpha) / (sum(self.uniDict.values())+alpha*self.vocabnum)) + \
                            backoff[1] * math.log((self.biDict[sent[i] + '|' + sent[i-1]]+alpha) / (self.bi_preDict[sent[i-1]]+alpha*self.vocabnum)) + \
                            backoff[2] * math.log((self.triDict[sent[i] + '|' + sent[i-2] + ',' + sent[i-1]]+alpha) / (self.tri_preDict[sent[i-2] + ',' + sent[i-1]]+alpha*sum(self.tri_preDict.values())))
        return sentProb

    def knSmoothing(self, sent):
        sentProb = 0
        d = 0.75
        for i in range(len(sent)):
            if i == 0:
                sentProb *= (self.uniDict[sent[i]]+0.001) / (sum(self.uniDict.values())+0.001*self.vocabnum)
            else:
                if self.uniDict[sent[i-1]] == 0:
                    sentProb *= 1 / sum(self.uniDict.values())
                else:
                    count1 = 0
                    count2 = 0
                    for key, value in self.biDict.items():
                        if key.split('|')[0] == sent[i] and value > 0:
                            count1 += 1
                        if key.split('|')[1] == sent[i-1] and value > 0:
                            count2 += 1
                    contProb = count1 / self.bicount
                    lbda = d / self.uniDict[sent[i-1]] * count2
                    sentProb += max(self.biDict[sent[i]+'|'+sent[i-1]]-d, 0.001) / self.uniDict[sent[i-1]] + lbda * contProb
        return sentProb

    def development(self):
        alphas = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        backoffs = [[0.2, 0.4, 0.4], [0.2, 0.3, 0.5], [0.1, 0.4, 0.5], [0.1, 0.3, 0.6], [0.1, 0.2, 0.7]]
        with codecs.open('../taskA/dev.tsv', 'r', encoding='utf_8') as f:
            devdata = f.readlines()
        f.close()
        ndev = len(devdata)
        summary = []
        for alpha in alphas:
            for backoff in backoffs:
                accuracy = 0
                for line in devdata:
                    row = line.strip('[\n.,?!]').split('\t')
                    correct = 1 - int(row[1])
                    prob1 = self.NgramVal(row[2].split(), alpha, backoff)
                    prob2 = self.NgramVal(row[3].split(), alpha, backoff)
                    if (correct == 0 and prob1 > prob2) or (correct == 1 and prob1 < prob2):
                        accuracy += 1
                summary.append({
                    "alpha": alpha,
                    "backoff": backoff,
                    "accuracy": accuracy/ndev,
                })
        for ele in summary:
            print(ele)

    def test(self):
        accuracy = 0
        numofline = 0
        with codecs.open('../taskA/dev.tsv', 'r', encoding='utf_8') as f:
            for line in f:
                numofline += 1
                row = line.strip('[\n.,?!]').split('\t')
                correct = 1 - int(row[1])
                prob1 = self.NgramVal(row[2].split(), 0.001, [0.2, 0.4, 0.4])
                prob2 = self.NgramVal(row[3].split(), 0.001, [0.2, 0.4, 0.4])
                # prob1 = self.knSmoothing(row[2].split())
                # prob2 = self.knSmoothing(row[3].split())
                if (correct == 0 and prob1 > prob2) or (correct == 1 and prob1 < prob2):
                    accuracy += 1
                else:
                    self.wrong.append(row[0])
                # if numofline % 100 == 0:
                    # print(numofline, accuracy/numofline)
        f.close()
        return accuracy / numofline


if __name__ == '__main__':
    # 朴实的方法， 稍微改一下第一次作业的代码，在这个数据集上做trigram
    Ngram = NgramModel()
    Ngram.dataloader()
    print(Ngram.ntrain)
    start = time.time()
    # Ngram.development()
    res = Ngram.test()
    print(res)
    end = time.time()
    print("total time: %.2f" % (end-start))
