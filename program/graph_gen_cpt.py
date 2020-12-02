from collections import defaultdict
import json
import numpy as np
import networkx as nx


def remove_useless():
    # 将非英文的关系从conceptnet中删除
    with open("../dataset/conceptnet-assertions-5.6.0.csv", 'r', encoding='utf8') as csvfile:
        file_object = open('../dataset/rel_base.txt', 'w', encoding='utf8')
        i = 0
        for line in csvfile:
            newline = line.split('\t')
            if newline[2].startswith('/c/en/') and newline[3].startswith('/c/en/'):
                rel = newline[1].lower().replace('/r/', '')
                head = newline[2].lower().replace('/c/en/', '').split('/')[0]
                tail = newline[3].lower().replace('/c/en/', '').split('/')[0]
                weight = line[line.rfind('weight')+9:-2]
                if head != weight:
                    newline2 = rel + '\t' + head + '\t' + tail + '\t' + weight + '\n'
                    file_object.write(newline2)

            i += 1
            if i % 10000 == 0:
                print(i)


def merge_relations():
    # 合并关系，并删除dbpedia
    dic = {'antonym': 'antonym', 'atlocation': 'atlocation', 'capableof': 'capableof',
           'causes': 'causes', 'causesdesire': 'causes', 'createdby': 'createdby',
           'definedas': 'isa', 'derivedfrom': 'causes', 'desires': 'desires',
           'distinctfrom': 'antonym', 'entails': 'hassubevent', 'etymologicallyderivedfrom': 'causes',
           'etymologicallyrelatedto': 'relatedto', 'formof': 'formof', 'hasa': 'partof',
           'hascontext': 'hascontext', 'hasfirstsubevent': 'hassubevent', 'haslastsubevent': 'hassubevent',
           'hasprerequisite': 'hassubevent', 'hasproperty': 'hasproperty', 'hassubevent': 'hassubevent',
           'instanceof': 'isa', 'isa': 'isa', 'locatednear': 'atlocation', 'madeof': 'madeof', 
           'mannerof': 'hassubevent', 'motivatedbygoal': 'causes', 'notcapableof': 'notcapableof',
           'notdesires': 'notdesires', 'nothasproperty': 'nothasproperty', 'partof': 'partof',
           'receivesaction': 'receivesaction', 'relatedto': 'relatedto', 'similarto': 'relatedto',
           'symbolof': 'relatedto', 'synonym': 'relatedto', 'usedfor': 'usedfor',
           # 'dbpedia/capital': 'relatedto', 'dbpedia/field': 'relatedto', 'dbpedia/genre': 'relatedto',
           # 'dbpedia/genus': 'relatedto', 'dbpedia/influencedby': 'relatedto', 'dbpedia/knownfor': 'relatedto',
           # 'dbpedia/language': 'relatedto', 'dbpedia/leader': 'relatedto', 'dbpedia/occupation': 'relatedto',
           # 'dbpedia/product': 'relatedto'}
           }

    with open("../dataset/rel_base.txt", 'r', encoding='utf8') as rel_file:
        file_object = open('../dataset/rel_base_en.txt', 'w', encoding='utf8')
        for line in rel_file:
            newline = line.split('\t')
            if newline[0].startswith('dbpedia') or newline[0].startswith('formof') or \
                    newline[0].startswith('nothasproperty') or newline[1] == newline[2]:
                continue
            newline[0] = dic[newline[0]]
            if newline[0].startswith('hasa') or newline[0].startswith('motivatedbygoal'):
                newline = newline[0] + '\t' + newline[2] + '\t' + newline[1] + '\t' + newline[3]
            else:
                newline = newline[0] + '\t' + newline[1] + '\t' + newline[2] + '\t' + newline[3]
            file_object.write(newline)


def create_index():
    # 获取知识库中所有的relation、entity并建立索引
    with open('../dataset/rel_base_en.txt', 'r', encoding='utf8') as file:
        entity = defaultdict(int)
        relation = defaultdict(int)
        triple2id = defaultdict(int)
        for line in file:
            newline = line.split('\t')
            relation[newline[0]] += 1
            entity[newline[1]] += 1
            entity[newline[2]] += 1
            triple2id[(newline[0], newline[1], newline[2])] = len(triple2id.items())
    print("Record Data... Done.")

    # 建立relation索引
    relation2id = defaultdict(int)
    file.close()
    f_rel = open('../dataset/relation2id.txt', 'w', encoding='utf8')
    for idx, rel in enumerate(list(relation.keys())):
        relation2id[rel] = idx
        f_rel.write(rel + '\t' + str(idx) + '\n')
    print('Create index for relation... Done.')

    # 建立entity索引
    entity2id = defaultdict(int)
    f_ent = open('../dataset/entity2id.txt', 'w', encoding='utf8')
    for idx, ent in enumerate(list(entity.keys())):
        entity2id[ent] = idx
        f_ent.write(ent + '\t' + str(idx) + '\n')
    print('Create index for entity... Done.')

    # 建立三元组关系索引
    f_tri = open('../dataset/train2id.txt', 'w', encoding='utf8')
    for triple, idx in triple2id.items():
        f_tri.write(str(entity2id[triple[1]]) + '\t' + str(entity2id[triple[2]]) + '\t' + str(
            relation2id[triple[0]]) + '\n')
    f_tri.close()
    print('Create index for triple... Done.')


def gen_graph():
    # 获取entity索引
    entity2id = defaultdict(int)
    id2entity = defaultdict(str)
    entity_file = open('../dataset/entity2id.txt', 'r', encoding='utf8').readlines()
    for line in entity_file:
        entity, idx = line.strip('\n').split('\t')
        entity2id[entity] = int(idx)
        id2entity[int(idx)] = entity

    # 获取relation索引
    relation2id = defaultdict(int)
    id2relation = defaultdict(str)
    relation_file = open('../dataset/relation2id.txt', 'r', encoding='utf8').readlines()
    for line in relation_file:
        rel, idx = line.strip('\n').split('\t')
        relation2id[rel] = int(idx)
        id2relation[int(idx)] = rel

    # 建立三元组关系的图结构
    graph = nx.MultiDiGraph()
    with open('../dataset/rel_base_en.txt', 'r', encoding='utf8') as file:
        lines = file.readlines()

        for line in lines:
            newline = line.strip().split('\t')
            rel = relation2id[newline[0]]
            head = entity2id[newline[1]]
            tail = entity2id[newline[2]]
            weight = float(newline[3])

            graph.add_edge(head, tail, rel=rel, weight=weight)
            graph.add_edge(tail, head, rel=rel+len(relation2id), weight=weight)

    nx.write_gpickle(graph, '../dataset/rel_base_en.graph')
    print('Build Graph... Done.')


def convert_transe_to_npy():
    # 保存TransE预训练好的entity以及relation的embedding初始值
    transe_file = '../dataset/emb_init/transe.sgd.vec.json'
    with open(transe_file, 'r') as file:
        transe_emb = json.load(file)

    ent_emb, rel_emb = transe_emb['ent_embeddings'], transe_emb['rel_embeddings']
    ent_embs = np.array(ent_emb, dtype="float32")
    rel_embs = np.array(rel_emb, dtype="float32")

    np.save('../dataset/emb_init/transe.sgd.ent.npy', ent_embs)
    np.save('../dataset/emb_init/transe.sgd.rel.npy', rel_embs)


if __name__ == '__main__':
    """
    rel_dict = defaultdict(int)   # 原本一共47种关系, 缩减为17种

    for line in open('../dataset/rel_base_en.txt', 'r', encoding='utf_8'):
        newline = line.split('\t')
        rel_dict[newline[0]] += 1
    for rel in rel_dict.items():
        print(rel)
    """

    remove_useless()
    merge_relations()
    create_index()
    gen_graph()
    # convert_transe_to_npy()

