from relation_extraction import triple_scoring
from relation_extraction import id2entity, id2relation
import pickle
import re


# 采用固定的规则将关系三元组转化为自然语言
def fixed_rules(sent):
    sent = re.sub(r"atlocation", "is at location of", sent)
    sent = re.sub(r"relatedto", "is related to", sent)
    sent = re.sub(r"notcapableof", "is not capable of", sent)
    sent = re.sub(r"capableof", "is capable of", sent)
    sent = re.sub(r"madeof", "is made of", sent)
    sent = re.sub(r"antonym", "is antonym of", sent)
    sent = re.sub(r"hasproperty", "has property of", sent)
    sent = re.sub(r"partof", "is part of", sent)
    sent = re.sub(r"isa", "is a", sent)
    sent = re.sub(r"hascontext", "has", sent)
    sent = re.sub(r"createdby", "is created by", sent)
    sent = re.sub(r"usedfor", "is used for", sent)
    sent = re.sub(r"hassubevent", "causes", sent)
    sent = re.sub(r"receivesaction", "is caused by", sent)
    sent = re.sub(r"notdesires", "does not desire", sent)
    return sent


# 为每条语句获取知识
def create_new_exp(exp):
    new_exp = []
    score_dict = {}
    for rel in exp:
        if rel is None or len(rel) == 0:
            return []
        else:
            relations = rel["pf_res"]
            if len(relations) == 0:
                return []
            for relation in relations:
                score = 1
                exp = ""
                for i in range(len(relation["rel"])):
                    rel = relation["rel"][i][0]
                    head, tail = relation["path"][i], relation["path"][i+1]
                    score = score * triple_scoring(head, rel, tail)
                    if id2relation[rel] == "hascontext" or id2relation[rel] == "hascontext*":
                        continue

                    if id2entity[tail] == "slang" or id2entity[head] == 'slang':
                        continue

                    if rel >= 17:
                        sub_exp = id2entity[tail] + ' ' + id2relation[rel - 17] + ' ' + id2entity[head]
                    else:
                        sub_exp = id2entity[head] + ' ' + id2relation[rel] + ' ' + id2entity[tail]
                    exp = exp + ' ' + sub_exp
                score_dict.update({exp: score})
    scorelst = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    scorelst = [x for x in scorelst if len(x[0].split()) >= 3]
    # print(scorelst)
    for element in scorelst[:3]:
        assert "*" not in element[0]
        new_exp.append(fixed_rules(element[0]))

    return new_exp


# 为原始数据集增加新的外部知识
def adding_explanations(filename):
    paths = pickle.load(open("../dataset/" + filename + "_path.2.pf.pruned.pickle", 'rb'))
    index = 1
    explanations = []
    for i in range(0, len(paths)-1, 2):
        path_form = {"data_id": index, "path1": paths[i], "path2": paths[i+1]}
        explanations.append(path_form)
        index += 1
    datafile = open("../dataset/" + filename + "_bert.txt", "r", encoding="utf_8")
    data = datafile.readlines()
    print(len(data), len(explanations))

    newdata = []
    # """
    for i in range(len(data)):
        line = eval(data[i])
        statement1, statement2 = line["statement1"], line["statement2"]
        exp1, exp2 = explanations[i]["path1"], explanations[i]["path2"]
        new_exp1, new_exp2 = create_new_exp(exp1), create_new_exp(exp2)
        print(new_exp1, new_exp2)
        assert len(new_exp1) <= 3
        assert len(new_exp2) <= 3
        for new_exp in new_exp1:
            statement1 = new_exp + statement1
        for new_exp in new_exp2:
            statement2 = statement2 + ':' + new_exp
        # print(statement1, '|||', statement2)
        newdata.append({"data_id": line["data_id"], "correct": line["correct"],
                        "statement1": statement1, "statement2": statement2})

    with open("../dataset/bert_new_" + filename + ".txt", "w", encoding="utf_8") as file:
        for line in newdata:
            file.writelines(str(line) + '\n')
    # """


if __name__ == "__main__":

    adding_explanations("test")
    # adding_explanations("dev")
    adding_explanations("train")