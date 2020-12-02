import networkx as nx
import itertools
import json
from tqdm import tqdm
import pickle
from relation_extraction import id2entity, entity2id
from relation_extraction import cpnet_simple


# 建立无向图
def plain_graph_generation(qcs, acs, paths, rels):
    # 这部分代码我们参考了KagNet的做法, 他们建立了没有关系的无向图
    # 我们也采用这种方法

    graph = nx.Graph()
    for index, p in enumerate(paths):

        for c_index in range(len(p)-1):
            h = p[c_index]
            t = p[c_index+1]
            graph.add_edge(h, t, weight=1.0)

    for qc1, qc2 in list(itertools.combinations(qcs, 2)):
        if cpnet_simple.has_edge(qc1, qc2):
            graph.add_edge(qc1, qc2, weight=1.0)

    for ac1, ac2 in list(itertools.combinations(acs, 2)):
        if cpnet_simple.has_edge(ac1, ac2):
            graph.add_edge(ac1, ac2, weight=1.0)

    if len(qcs) == 0:
        qcs.append(-1)

    if len(acs) == 0:
        acs.append(-1)

    if len(paths) == 0:
        for qc in qcs:
            for ac in acs:
                graph.add_edge(qc, ac, rel=-1, weight=0.1)

    g = nx.convert_node_labels_to_integers(graph, label_attribute='cid')  # re-index
    g_str = json.dumps(nx.node_link_data(g))
    return g_str


def gen_graph(filename, k):
    with open("../dataset/" + filename + "_path." + str(k) + ".pf.pruned.pickle", "rb") as fi:
        pf_data = pickle.load(fi)
    with open("../dataset/diffwords_" + filename + ".mcp", "r") as f:
        mcp_data = json.load(f)

    final_text = ""
    assert len(pf_data) == len(mcp_data)
    for index, qa_pairs in tqdm(enumerate(pf_data), desc="Building Graphs", total=len(pf_data)):
        statement_paths = []
        statement_rel_list = []
        for qa_idx, qas in enumerate(qa_pairs):
            if qas["pf_res"] is None:
                cur_paths = []
                cur_rels = []
            else:
                cur_paths = [item["path"] for item in qas["pf_res"]]
                cur_rels = [item["rel"] for item in qas["pf_res"]]
            statement_paths.extend(cur_paths)
            statement_rel_list.extend(cur_rels)

        qcs = [entity2id[c] for c in mcp_data[index]["source"] if c in entity2id.keys()]
        acs = [entity2id[c] for c in mcp_data[index]["target"] if c in entity2id.keys()]

        gstr = plain_graph_generation(qcs=qcs, acs=acs, paths=statement_paths, rels=statement_rel_list)
        final_text += gstr + "\n"
    with open("../dataset/" + filename + "_path." + str(k) + ".pf.pruned.pnxg", 'w') as fw:
        fw.write(final_text)
    fw.close()


if __name__ == '__main__':
    gen_graph('train', 2)
    gen_graph('test', 2)
    # gen_graph('dev', 1)
