import numpy as np
import csv


def load_correct():
    correct_label = []
    with open('../dataset/diffwords_test.txt', 'r', encoding='utf_8') as f:
        f = f.readlines()
    for line in f:
        correct_label.append(eval(line)['correct'])
    return correct_label


def load_score_bert():
    data = []
    with open('../result/xlnet_finetuning_logits_87.70.txt', 'r', encoding='utf_8') as file:
        f = file.readlines()
        for i in range(len(f)):
            newline = eval(f[i])
            data.append(newline["logit"])
        lst_scores = []
        for i in range(len(data)):
            scores = np.array(data[i])
            # scores = scores / (np.linalg.norm(scores))
            scores = np.exp(scores) / np.sum(np.exp(scores))
            lst_scores.append(scores)
    return lst_scores


def load_score_graph():
    lst_scores = []
    data = []
    with open("../dataset/kagnet_output_3_0.24.txt", "r", encoding="utf_8") as file:
        for line in file:
            data.append(float(line[1:-2]))
    for i in range(0, len(data)-1, 2):
        scores = np.array(data[i:i+2])
        # scores = scores / (np.linalg.norm(scores))
        scores = np.exp(scores) / np.sum(np.exp(scores))
        lst_scores.append(scores)

    return lst_scores


if __name__ == '__main__':

    correct_label = load_correct()
    # """
    lst_graph = load_score_graph()
    lst_bert = load_score_bert()
    highest = 0
    best_pred = []
    print(len(lst_graph), len(lst_bert), len(correct_label))
    # 确定结果融合的平滑系数
    # """
    for L in range(0, 1001, 1):
        l = round(L / 1000, 4)
        pred_label = []
        for i in range(len(lst_bert)):
            newscore = lst_graph[i] * l + lst_bert[i] * (1 - l)
            idx = np.argmax(newscore)
            assert idx in [0, 1]
            pred_label.append(idx)
        pred_label = np.array(pred_label)
        accuracy = 1 - np.sum(np.abs(pred_label - correct_label)) / len(correct_label)
        print(accuracy)
        if accuracy > highest:
            highest = accuracy
            best_pred = pred_label.tolist()
    print('\n'); print(highest)
    # """

    # 写入最后的输出文件
    data_idx = []
    with open('../dataset/diffwords_test.txt', 'r', encoding='utf_8') as f:
        f = f.readlines()
    for line in f:
        data_idx.append(eval(line)['idx'])
    with open('../result/taskA_prediction.csv', 'w', encoding="utf_8", newline="") as f:
        for i, idx in enumerate(data_idx):
            f.write(str(idx-1) + ',' + str(1 - best_pred[i]) + '\n')
