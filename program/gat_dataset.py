import torch
import torch.utils.data as data
import numpy as np
import json
from tqdm import tqdm
import timeit
import pickle
import os
import dgl
import networkx as nx
import random


class data_with_graphs(data.Dataset):

    def __init__(self, statement_json_file, graph_ngx_file, num_choice=2
                 , start=0, end=None, reload=True):

        self.qids = []
        self.correct_labels = []

        statement_json_data = []
        print("loading statements from %s" % statement_json_file)
        with open(statement_json_file, "r") as fp:
            for line in fp.readlines():
                statement_data = json.loads(line.strip())
                statement_json_data.append(statement_data)
        print("Num of statements: %d" % len(statement_json_data))

        self.qa_text = []
        for question_id in range(len(statement_json_data)):
            qa_text_cur = []
            self.qids.append([statement_json_data[question_id]["id"]])
            for k, s in enumerate(statement_json_data[question_id]["statements"]):
                assert len(statement_json_data[question_id]["statements"]) == num_choice  # 一条样本有一对statements
                qa_text_cur.append((s["statement"], s['label']))
                if s["label"] is True:
                    self.correct_labels.append(k)  # the truth id [0,1]

            self.qa_text.append(qa_text_cur)
        print("Num of qa_texts: %d" % len(self.qa_text))

        self.nxgs = []
        self.dgs = []
        start_time = timeit.default_timer()
        print("loading paths from %s" % graph_ngx_file)
        with open(graph_ngx_file, 'r') as fr:
            for line in fr.readlines():
                line = line.strip()
                self.nxgs.append(line)
        print('\t Done! Time: ', "{0:.2f} sec".format(float(timeit.default_timer() - start_time)))

        save_file = graph_ngx_file + ".dgl.pk"

        if reload and os.path.exists(save_file):
            import gc
            print("loading pickle for the dgl", save_file)
            start_time = timeit.default_timer()
            with open(save_file, 'rb') as handle:
                gc.disable()
                self.dgs = pickle.load(handle)
                gc.enable()
            print("finished loading in %.3f secs" % (float(timeit.default_timer() - start_time)))

        else:
            for index, nxg_str in tqdm(enumerate(self.nxgs), total=len(self.nxgs)):
                nxg = nx.node_link_graph(json.loads(nxg_str))
                dg = dgl.DGLGraph(multigraph=True)
                # dg.from_networkx(nxg, edge_attrs=["rel"])
                dg.from_networkx(nxg)
                cids = [nxg.nodes[n_id]['cid']+1 for n_id in range(len(dg))]

                dg.ndata.update({'cncpt_ids': torch.LongTensor(cids)})
                self.dgs.append(dg)

            save_file = graph_ngx_file + ".dgl.pk"
            print("saving pickle for the dgl", save_file)
            with open(save_file, 'wb') as handle:
                pickle.dump(self.dgs, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.nxgs = list(zip(*(iter(self.nxgs),) * num_choice))
        self.dgs = list(zip(*(iter(self.dgs),) * num_choice))
        print("loading graphs...done")

        # slicing dataset
        self.correct_labels = self.correct_labels[start:end]
        self.qids = self.qids[start:end]
        self.nxgs = self.nxgs[start:end]
        self.dgs = self.dgs[start:end]

        assert len(self.correct_labels) == len(self.qids)
        self.n_samples = len(self.correct_labels)

    def slice(self, start=0, end=None):
        # slicing dataset
        all_lists = list(zip(self.correct_labels, self.qids, self.nxgs, self.dgs))
        random.shuffle(all_lists)
        self.correct_labels, self.qids, self.nxgs, self.dgs = zip(*all_lists)

        self.correct_labels = self.correct_labels[start:end]
        self.qids = self.qids[start:end]
        self.nxgs = self.nxgs[start:end]
        self.dgs = self.dgs[start:end]
        assert len(self.correct_labels) == len(self.qids)
        self.n_samples = len(self.correct_labels)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return torch.Tensor([self.correct_labels[index]]), self.dgs[index],  self.qa_text[index]


def collate_csqa_graphs(samples):
    # The input `samples` is a list of pairs
    #  (graph, label, qid, aid, sentv).
    correct_labels, graph_data, qa_text = map(list, zip(*samples))

    flat_graph_data = []
    for gd in graph_data:
        flat_graph_data.extend(gd)

    concept_mapping_dicts = []
    acc_start = 0
    for k, g in enumerate(flat_graph_data):
        concept_mapping_dict = {}
        for index, cncpt_id in enumerate(g.ndata['cncpt_ids']):
            concept_mapping_dict[int(cncpt_id)] = acc_start + index

        acc_start += len(g.nodes())
        concept_mapping_dicts.append(concept_mapping_dict)

    batched_graph = dgl.batch(flat_graph_data)
    return torch.Tensor([[i] for i in correct_labels]), batched_graph,  concept_mapping_dicts


if __name__ == "__main__":
    train_set = data_with_graphs("../dataset/train.statements",
                                           "../dataset/train_path.2.pf.pruned.pnxg",
                                           num_choice=2, reload=False, start=0, end=None)
    print(len(train_set))
