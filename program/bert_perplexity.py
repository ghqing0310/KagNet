import numpy as np
from transformers import BertForMaskedLM, BertTokenizer
import torch


def tsv2txt(filename):
    with open('../taskA/' + filename + '.tsv', 'r', encoding='utf_8') as file:
        f = open('../dataset/bert.' + filename + '.txt', 'w', encoding='utf_8')
        correct_label = []
        for line in file:
            row = line.strip("[\n.,?!']").lower().split('\t')
            correct = 1 - int(row[1])
            correct_label.append(correct)
            sent1, sent2 = row[2].strip('.'), row[3].strip('.')
            f.writelines(sent1 + '\n')
            f.writelines(sent2 + '\n')
    return correct_label


def accuracy(filename):
    with open('../dataset/bert.output.' + filename + '.txt') as file:
        pred = []
        for line in file:
            start = line.find(':') + 1
            pred.append(float(line[start:-2]))
    pred_label = []
    for i in range(0, len(pred)-1, 2):
        pred_label.append(pred[i:i+2].index(min(pred[i:i+2])))
    return pred_label


def sent_score(sentence):
    # 依次遮盖每一个token，利用其它部分计算当前token出现概率
    # 将概率累乘获得整句句子概率， 以及句子困惑度
    tokenize_input = ['[CLS]'] + tokenizer.tokenize(sentence) + ['[SEP]']
    tokenize_id_input = tokenizer.convert_tokens_to_ids(tokenize_input)
    tensor_input = torch.tensor([tokenize_id_input])
    sentence_scores = 1

    for i, word in enumerate(tokenize_input):
        if tokenize_input[i] == '[CLS]' or tokenize_input[i] == '[SEP]':
            continue
        tmp = tokenize_input[i]
        tokenize_input[i] = '[MASK]'
        mask_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        prediction_score = model(mask_input, masked_lm_labels=tensor_input)[1].data.numpy()
        position_score = np.exp(prediction_score[0][i][tokenize_id_input[i]]) / np.sum(np.exp(prediction_score[0][i]))
        # print(position_score)
        sentence_scores *= 1 / position_score
        tokenize_input[i] = tmp

    return np.power(sentence_scores, 1 / (len(tokenize_input)-2))


def perplexity(filename):
    file = open('../dataset/bert.' + filename + '.txt', 'r', encoding='utf_8')
    all_scores = []
    i = 0
    for line in file:
        print(i)
        i += 1
        score = sent_score(line.strip('\n'))
        all_scores.append(score)
    with open("../dataset/bert.output.torch.test.txt", "w", encoding="utf_8") as file:
        for element in all_scores:
            file.writelines(str(element) + '\n')
    file.close()
    pred_label = []
    for i in range(0, len(all_scores)-1, 2):
        pred_label.append(all_scores[i] > all_scores[i+1])
    return pred_label


if __name__ == '__main__':
    # 导入模型
    with torch.no_grad():
        model = BertForMaskedLM.from_pretrained('bert-large-uncased')
        model.eval()
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    # 调整数据集格式
    correct = tsv2txt('test')
    # 困惑度计算
    pred = perplexity('test')
    # 输出正确率
    print(1 - np.sum(np.abs(np.array(correct) - np.array(pred))) / len(pred))
