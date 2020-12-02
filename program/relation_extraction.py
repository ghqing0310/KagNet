from collections import defaultdict
from scipy import spatial
import numpy as np
import networkx as nx
import time
import json
import pickle
from sklearn.externals import joblib


relation2id, entity2id, id2relation, id2entity = {}, {}, {}, {}
with open('../dataset/relation2id.txt', 'r', encoding='utf_8') as file:
    file.readline()
    f = file.readlines()
    for element in f:
        element2 = element.strip('\n').split()
        relation2id[element2[0]] = int(element2[1])
        id2relation[int(element2[1])] = element2[0]
    # 为逆关系建立索引
    num_rel = len(id2relation.items())
    for i in range(num_rel):
        id2relation.update({i+num_rel: id2relation[i]+'*'})
print('relation2id loaded.')

with open('../dataset/entity2id.txt', 'r', encoding='utf_8') as file:
    f = file.readlines()
    for element in f:
        element2 = element.strip('\n').split()
        entity2id[element2[0]] = int(element2[1])
        id2entity[int(element2[1])] = element2[0]
print('entity2id loaded.')
entity_emb = np.load('../dataset/emb_init/transe.sgd.ent.npy')
relation_emb = np.load('../dataset/emb_init/transe.sgd.rel.npy')
print('initial embeddings loaded.')
# print(relation_emb.shape)
stop_words = ['in', 'out', 'of', 'to', 'on', 'at', 'off', 'as', 'up', 'down', 'around', 'into',
              'is', 'be', 'am', 'are', 'was', 'were', 'been', 'should', 'need',
              'he', 'she', 'they', 'i', 'you',
              'his', 'her', 'their', 'my', 'me', 'your', 'we', 'us',
              'that', 'this', 'those', 'these', 'there', 'here']


# method 1
class RelExtractor(object):
    def __init__(self):
        self.train_file = '../dataset/diffwords_train.txt'
        self.test_file = '../dataset/diffwords_test.txt'
        self.rel_base_file = '../dataset/rel_base_en.txt'
        self.rel_base = []

    def load_train_data(self):
        train_data = []
        f = open(self.train_file, 'r', encoding='utf_8')
        for line in f.readlines():
            train_data.append(eval(line))
        return train_data

    def load_test_data(self):
        test_data = []
        f = open(self.test_file, 'r', encoding='utf_8')
        for line in f.readlines():
            test_data.append(eval(line))
        return test_data

    def load_kg(self):
        rel_file = open(self.rel_base_file, 'r', encoding='utf_8')
        self.rel_base = rel_file.readlines()

    def extract_between_entity(self, diffword, sameword):
        graph1 = defaultdict()
        graph2 = defaultdict()
        paths = {'ac': diffword, 'qc': sameword, 'pf_res': []}

        for line in self.rel_base:
            line = tuple(line.strip('\n').split('\t'))

            if diffword == line[1]:
                graph1[line[2]] = line[0] if line[2] not in graph1.keys() else graph1[line[2]] + '/' + line[0]
            elif diffword == line[2]:
                graph1[line[1]] = line[0] + '*'if line[1] not in graph1.keys() else graph1[line[1]] + '/' + line[0] + '*'

            if sameword == line[1]:
                graph2[line[2]] = line[0] + '*' if line[2] not in graph2.keys() else graph2[line[2]] + '/' + line[0]
            elif sameword == line[2]:
                graph2[line[1]] = line[0] if line[1] not in graph2.keys() else graph2[line[1]] + '/' + line[0] + '*'

        for entity in graph1.keys():
            if entity == sameword:
                rels = list(set(graph1[entity].split('/')))
                paths['pf_res'].append({'path': [diffword, sameword], 'rel': [rels]})
            else:
                for entity2 in graph2.keys():
                    if entity == entity2:
                        rels1 = list(set(graph1[entity].split('/')))
                        rels2 = list(set(graph2[entity].split('/')))
                        path = {'path': [diffword, entity, sameword], 'rel': [rels1, rels2]}
                        # if self.prune_or_not(path['path'], path['rel']) == 1:
                        #    continue
                        paths['pf_res'].append(path)

        return [paths]

    def triple_scoring(self, head, rel, tail):
        if rel in relation2id.keys():   # 无 inverse
            head_idx, tail_idx = entity2id[head], entity2id[tail]
            rel_idx = relation2id[rel]
        else:   # inverse
            head_idx, tail_idx = entity2id[tail], entity2id[head]
            rel_idx = relation2id[rel[:-1]]
        score = max(0, (1 + 1 - spatial.distance.cosine(relation_emb[rel_idx], entity_emb[tail_idx] -
                                                        entity_emb[head_idx])) / 2)
        return score

    def path_scoring(self, entitys, rels):
        score = 1
        score_of_path = []
        for i in range(len(rels)):
            sub_rels = rels[i]
            sub_path = entitys[i:i+2]
            sub_path_score = 0
            for rel in sub_rels:
                sub_path_score = max(sub_path_score, self.triple_scoring(sub_path[0], rel, sub_path[1]))
            score_of_path.append(round(sub_path_score, 4))
            score *= sub_path_score

        return round(score, 4)

    def prune_or_not(self, entitys, rels):
        threshold = 0.2

        score = self.path_scoring(entitys, rels)
        return (score < threshold)


# method 2
cpnet = nx.read_gpickle('../dataset/rel_base_en.graph')
cpnet_simple = nx.Graph()
for u, v, data in cpnet.edges(data=True):
    w = data['weight'] if 'weight' in data else 1.0
    if cpnet_simple.has_edge(u, v):
        cpnet_simple[u][v]['weight'] += w
    else:
        cpnet_simple.add_edge(u, v, weight=w)
print('cpt graph loaded.')


def get_edge(src_concept, tgt_concept):
    # 返回两个实体间连接的关系
    rel_list = cpnet[src_concept][tgt_concept]
    #rels = [id2relation[rel_list[ele]['rel']] for ele in rel_list]  # 返回rel名
    rels = [rel_list[ele]['rel'] for ele in rel_list] # 返回rel序号
    return list(set(rels))


def extract_between_entity(head, tail, k):
    # 抽取两个实体间存在的关系
    if head not in entity2id.keys() or tail not in entity2id.keys():
        return
    head_idx = entity2id[head]
    tail_idx = entity2id[tail]
    assert head_idx in cpnet_simple.nodes()
    assert tail_idx in cpnet_simple.nodes()
    paths_lst = []
    for path_len in range(1, k+1):
        for path in nx.all_simple_paths(cpnet_simple, head_idx, tail_idx, path_len):
            if path not in paths_lst:

                # path = [id2entity[x] for x in path]     # 返回entity名称

                paths_lst.append(path)
    # print(paths_lst)
    pf_res = []
    for path in paths_lst:
        rels = []
        for i in range(len(path)-1):

            head = path[i]
            tail = path[i+1]
            # head = entity2id[path[i]]
            # tail = entity2id[path[i+1]]

            rel = get_edge(head, tail)
            rels.append(rel)
        pf_res.append({'path': path, 'rel': rels})

    return pf_res


def extract_relation(filename, k):
    paths = []
    # with open('../dataset/2.txt', 'r', encoding='utf_8') as file:
    with open('../dataset/diffwords_' + filename + '.txt', 'r', encoding='utf_8') as file:
        i = 0
        for line in file.readlines():
            if i % 100 == 0:
                print(i)
            i += 1
            newline = eval(line.strip('\n'))
            samewords = newline['samewords']
            samewords = [x for x in samewords if x not in stop_words]
            diff_s1, diff_s2 = newline['diffwords'][0], newline['diffwords'][1]

            if len(samewords) == 0:
                # print(line)
                paths.append([{'ac': 'unk', 'qc': 'unk', 'pf_res': []}])
                paths.append([{'ac': 'unk', 'qc': 'unk', 'pf_res': []}])
                continue
            paths_s1, paths_s2 = [], []
            if len(diff_s1) == 0:
                # print(line)
                paths.append([{'ac': 'unk', 'qc': 'unk', 'pf_res': []}])
            else:
                for diffword in diff_s1:
                    for sameword in samewords:
                        paths_s1.append(
                            {'ac': diffword, 'qc': sameword, 'pf_res': extract_between_entity(diffword, sameword, k)})
                paths.append(paths_s1)
            if len(diff_s2) == 0:
                # print(line)
                paths.append([{'ac': 'unk', 'qc': 'unk', 'pf_res': []}])
            else:
                for diffword in diff_s2:
                    for sameword in samewords:
                        paths_s2.append(
                            {'ac': diffword, 'qc': sameword, 'pf_res': extract_between_entity(diffword, sameword, k)})
                paths.append(paths_s2)
    print('Extract Relations for ' + filename + '... Done. ')

    # with open('../dataset/' + filename + '_path.' + str(k) + '.pf.pickle', 'wb') as file_path:
    file_path = '../dataset/' + filename + '_path.' + str(k) + '.pf.pickle'
    joblib.dump(paths, filename=file_path)


def triple_scoring(head_idx, rel_idx, tail_idx):
    # 采用余弦距离来计算一个三元组关系的分数
    if rel_idx >= 17:
        rel_idx = rel_idx - 17
        tmp = tail_idx
        tail_idx = head_idx
        head_idx = tmp
    assert rel_idx < 17
    score = max(0, (1 + 1 - spatial.distance.cosine(relation_emb[rel_idx], entity_emb[tail_idx] -
                                                    entity_emb[head_idx])) / 2)
    return score


def path_scoring(filename, k):
    # 对已搜索到的关系进行打分
    threshold = 0.16
    path_count = 0
    path_to_be_pruned = 0
    with open("../dataset/" + filename + "_path." + str(k) + ".pf" + ".pickle", 'rb') as f:
        paths = pickle.load(f)
    print(len(paths))
    print("paths loaded")
    pf = []

    for one_pair in paths:
        score_one_pair = []

        for path_entity2entity in one_pair:
            score_entity2entity = []

            if path_entity2entity["pf_res"] is not None:
                path_count += len(path_entity2entity["pf_res"])

                for element in path_entity2entity["pf_res"]:
                    path, rel = element["path"], element["rel"]
                    score_of_one_path = 1
                    for i in range(len(path)-1):
                        head, tail = path[i], path[i+1]
                        rels = rel[i]
                        score_of_subpath = 0
                        for rel_ in rels:
                            score_of_subpath = max(score_of_subpath, triple_scoring(head, rel_, tail))
                        score_of_one_path *= score_of_subpath

                    score_entity2entity.append(score_of_one_path)
            else:
                score_entity2entity = []

            score_one_pair.append(score_entity2entity)

        pf.append(score_one_pair)
    print('total path count: %d' % path_count)
    with open('../dataset/' + filename + '_path.' + str(k) + '.score.pf.pickle', 'wb') as file:
        pickle.dump(pf, file, protocol=pickle.HIGHEST_PROTOCOL)


def path_pruning(filename, threshold, k):
    # 删除分数低于阈值的关系
    paths = pickle.load(open('../dataset/' + filename + '_path.' + str(k) + '.pf' + '.pickle', 'rb'))
    paths_scores = pickle.load(open('../dataset/' + filename + '_path.' + str(k) + '.score.pf.pickle', 'rb'))
    pruned_count = 0
    for pair_idx, one_pair in enumerate(paths):
        for path_idx, path_entity2entity in enumerate(one_pair):
            new_path_entity2entity = []
            if path_entity2entity['pf_res'] is not None:
                for idx, element in enumerate(path_entity2entity['pf_res']):
                    score = paths_scores[pair_idx][path_idx][idx]
                    if score > threshold:
                        new_path_entity2entity.append(element)
                    else:
                        pruned_count += 1
            paths[pair_idx][path_idx]["pf_res"] = new_path_entity2entity

    print('total pruned path count: %d' % pruned_count)
    with open("../dataset/" + filename + "_path." + str(k) + ".pf.pruned.pickle", 'wb') as fp:
        pickle.dump(paths, fp, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    # """
    extract_relation('train', 2)
    path_scoring('train', 2)
    path_pruning('train', 0.2, 2)

    #extract_relation('dev', 2)
    #path_scoring('dev', 2)
    #path_pruning('dev', 0.2, 2)

    extract_relation('test', 2)
    path_scoring('test', 2)
    path_pruning('test', 0.2, 2)
    # """
    # 测试
    # print(extract_between_entity('hospital', 'treatment', 2))
    # print(extract_between_entity('hospital', 'car_crash', 2))
    # print(extract_between_entity('hospital', 'crash', 2))
    # print(extract_between_entity('treatment', 'crash', 2))

