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


def load_embeddings(path):
    print("Loading glove concept embeddings with pooling:", path)
    concept_vec = np.load(path)
    print("done!")
    return concept_vec


class data_with_graphs_and_paths(data.Dataset):

    def __init__(self, statement_json_file, graph_ngx_file, pf_json_file, num_choice=2
                 , start=0, end=None, reload=True, cut_off=3):

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

        # load all qa and paths
        self.qa_pair_data = []
        self.cpt_path_data = []
        self.rel_path_data = []

        start_time = timeit.default_timer()
        print("loading paths from %s" % pf_json_file)
        with open(pf_json_file, 'rb') as handle:
            pf_json_data = pickle.load(handle)
        print('\t Done! Time: ', "{0:.2f} sec".format(float(timeit.default_timer() - start_time)))

        assert len(statement_json_data) * num_choice == len(pf_json_data)
        for s in tqdm(pf_json_data, desc="processing paths"):
            paths = []
            rels = []
            qa_pairs = list()
            for qas in s:
                pf_res = qas["pf_res"]
                if pf_res is not None:
                    for item in pf_res:
                        p = item["path"]
                        q = p[0] + 1
                        a = p[-1] + 1

                        if len(p) > cut_off:
                            continue  # cut off by length of concepts

                        # padding dummy concepts and relations
                        p = [n + 1 for n in p]
                        p.extend([0] * (cut_off - len(p)))  # padding

                        r = item["rel"]
                        for i_ in range(len(r)):
                            for j_ in range(len(r[i_])):
                                if r[i_][j_] - 17 in r[i_]:
                                    r[i_][j_] -= 17  # to delete realtedto* and antonym*

                        r = [n[0] + 1 for n in r]  # only pick the top relation when multiple ones are okay
                        r.extend([0] * (cut_off - len(r)))  # padding

                        assert len(p) == cut_off
                        paths.append(p)
                        rels.append(r)

                        if (q, a) not in qa_pairs:
                            qa_pairs.append((q, a))

            self.qa_pair_data.append(list(qa_pairs))
            self.cpt_path_data.append(paths)
            self.rel_path_data.append(rels)

        self.cpt_path_data = list(zip(*(iter(self.cpt_path_data),) * num_choice))
        self.rel_path_data = list(zip(*(iter(self.rel_path_data),) * num_choice))
        self.qa_pair_data = list(zip(*(iter(self.qa_pair_data),) * num_choice))

        # slicing dataset
        # self.statements = self.statements[start:end]
        self.correct_labels = self.correct_labels[start:end]
        self.qids = self.qids[start:end]
        self.nxgs = self.nxgs[start:end]
        self.dgs = self.dgs[start:end]

        assert len(self.correct_labels) == len(self.qids)
        self.n_samples = len(self.correct_labels)

    def slice(self, start=0, end=None):
        # slicing dataset
        # all_lists = list(zip(self.statements, self.correct_labels, self.qids, self.nxgs, self.dgs))
        all_lists = list(zip(self.correct_labels, self.qids, self.nxgs, self.dgs))
        random.shuffle(all_lists)
        # self.statements, self.correct_labels, self.qids, self.nxgs, self.dgs = zip(*all_lists)
        self.correct_labels, self.qids, self.nxgs, self.dgs = zip(*all_lists)

        # self.statements = self.statements[start:end]
        self.correct_labels = self.correct_labels[start:end]
        self.qids = self.qids[start:end]
        self.nxgs = self.nxgs[start:end]
        self.dgs = self.dgs[start:end]
        assert len(self.correct_labels) == len(self.qids)
        self.n_samples = len(self.correct_labels)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return torch.Tensor([self.correct_labels[index]]), self.dgs[index], \
               self.cpt_path_data[index], self.rel_path_data[index], self.qa_pair_data[index], self.qa_text[index]


def collate_csqa_graphs_and_paths(samples):
    # The input `samples` is a list of pairs
    #  (graph, label, qid, aid, sentv).
    correct_labels, graph_data, cpt_path_data, rel_path_data, qa_pair_data, qa_text = map(list, zip(*samples))

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
    return torch.Tensor([[i] for i in correct_labels]), batched_graph, cpt_path_data, rel_path_data, qa_pair_data, concept_mapping_dicts


if __name__ == "__main__":
    # 测试
    dev_set = data_with_graphs_and_paths("../dataset/test.statements",
                                         "../dataset/test_path.2.pf.pruned.pnxg",
                                         "../dataset/test_path.2.pf.pruned.pickle",
                                         num_choice=2, reload=False, cut_off=3, start=0, end=None)[:2]

    dataset_loader = data.DataLoader(dev_set, batch_size=1, num_workers=0, shuffle=False,
                                     collate_fn=collate_csqa_graphs_and_paths)
    for k, ele in enumerate(
            tqdm(dataset_loader, desc="Train Batch")):
        print(k)