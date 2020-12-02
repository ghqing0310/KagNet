import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np
from torch.nn import init
import torch.utils.data as data
from torch.autograd import Variable
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import json
from graph_models import KnowledgeAwareGraphNetworks, weight_init
from tqdm import tqdm
from graph_dataset import data_with_graphs_and_paths,  collate_csqa_graphs_and_paths
import copy
import random
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def load_embeddings(pretrain_embed_path):
    print("Loading glove concept embeddings with pooling:", pretrain_embed_path)
    concept_vec = np.load(pretrain_embed_path)
    print("done!")
    return concept_vec


def train_epoch_kag_netowrk(train_set, batch_size, optimizer, device, model, num_choice, loss_func):
    model.train()
    dataset_loader = data.DataLoader(train_set, batch_size=batch_size, num_workers=0, shuffle=False,
                                     collate_fn=collate_csqa_graphs_and_paths)
    bce_loss_func = nn.BCELoss()
    for k, (correct_labels, graphs, cpt_paths, rel_paths, qa_pairs, concept_mapping_dicts) in enumerate(
            tqdm(dataset_loader, desc="Train Batch")):
        # print(correct_labels)
        # print(graphs)
        # print(cpt_paths)
        # print(rel_paths)
        # print(qa_pairs)   数据对齐
        optimizer.zero_grad()
        correct_labels = correct_labels.to(device)
        graphs.ndata['cncpt_ids'] = graphs.ndata['cncpt_ids'].to(device)
        flat_qa_pairs = []
        flat_cpt_paths = []
        flat_rel_paths = []
        assert len(cpt_paths) == len(rel_paths) == len(qa_pairs)
        for i in range(len(cpt_paths)):
            # cur_statement = statements[i][0]  # num_choice statements
            cur_qa_pairs = qa_pairs[i]
            cur_cpt_paths = cpt_paths[i]
            cur_rel_paths = rel_paths[i]

            flat_qa_pairs.extend(cur_qa_pairs)
            flat_cpt_paths.extend(cur_cpt_paths)
            flat_rel_paths.extend(cur_rel_paths)

        flat_logits = model(flat_qa_pairs, flat_cpt_paths, flat_rel_paths, graphs, concept_mapping_dicts)
        y = torch.Tensor([1] * len(cpt_paths) * (num_choice - 1)).to(device)

        assert len(flat_logits) == len(flat_qa_pairs)
        x1 = []
        x2 = []

        for j, correct in enumerate(correct_labels):
            # for a particular qeustion
            for i in range(num_choice):
                cur_logit = flat_logits[j * num_choice + i]
                if i != correct[0]:  # for wrong answers
                    x2.append(cur_logit)
                else:  # for the correct answer
                    for _ in range(num_choice - 1):
                        x1.append(cur_logit)
        mrloss = loss_func(torch.cat(x1), torch.cat(x2), y)
        """
        cnt_correct = 0
        for j, correct in enumerate(correct_labels):
            # for a particular qeustion
            max_logit = None
            pred = 0
            for i in range(num_choice):
                cur_logit = flat_logits[j * num_choice + i]
                if max_logit is None:
                    max_logit = cur_logit
                    pred = i
                if max_logit < cur_logit:
                    max_logit = cur_logit
                    pred = i

            if correct[0] == pred:
                cnt_correct += 1
        acc = cnt_correct / len(correct_labels)
        print(acc)
        """
        mrloss.backward()
        optimizer.step()


def eval_kag_netowrk(eval_set, batch_size,  device, model, num_choice):
    model.eval()
    dataset_loader = data.DataLoader(eval_set, batch_size=batch_size, num_workers=0, shuffle=False,
                                     collate_fn=collate_csqa_graphs_and_paths)
    cnt_correct = 0
    total_logit = []
    for k, (correct_labels, graphs, cpt_paths, rel_paths, qa_pairs, concept_mapping_dicts) in enumerate(
            tqdm(dataset_loader, desc="Eval Batch")):

        correct_labels = correct_labels.to(device)
        graphs.ndata['cncpt_ids'] = graphs.ndata['cncpt_ids'].to(device)

        flat_qa_pairs = []
        flat_cpt_paths = []
        flat_rel_paths = []
        assert len(cpt_paths) == len(rel_paths) == len(qa_pairs)
        for i in range(len(cpt_paths)):

            cur_qa_pairs = qa_pairs[i]
            cur_cpt_paths = cpt_paths[i]
            cur_rel_paths = rel_paths[i]

            flat_qa_pairs.extend(cur_qa_pairs)
            flat_cpt_paths.extend(cur_cpt_paths)
            flat_rel_paths.extend(cur_rel_paths)

        flat_logits = model(flat_qa_pairs, flat_cpt_paths, flat_rel_paths, graphs,
                            concept_mapping_dicts)
        # print(len(flat_logits.data.numpy().tolist()))
        total_logit = total_logit + flat_logits.data.cpu().numpy().tolist()

        for j, correct in enumerate(correct_labels):
            # for a particular qeustion
            max_logit = None
            pred = 0
            for i in range(num_choice):
                cur_logit = flat_logits[j * num_choice + i]
                if max_logit is None:
                    max_logit = cur_logit
                    pred = i
                if max_logit < cur_logit:
                    max_logit = cur_logit
                    pred = i

            if correct[0] == pred:
                cnt_correct += 1
    acc = cnt_correct / len(eval_set)
    with open('../result/kagnet_output.txt', 'w', encoding='utf_8') as file:
        for element in total_logit:
            file.writelines(str(element) + '\n')
    file.close()
    return acc


def train():
    # 准备TransE预训练好的embedding
    pretrain_cpt_emd_path = "../dataset/emb_init/transe.sgd.ent.npy"
    pretrain_rel_emd_path = "../dataset/emb_init/transe.sgd.rel.npy"

    pretrained_concept_emd = load_embeddings(pretrain_cpt_emd_path)
    pretrained_relation_emd = load_embeddings(pretrain_rel_emd_path)
    print("pretrained_entity_emb.shape:", pretrained_concept_emd.shape)
    print("pretrained_relation_emb.shape:", pretrained_relation_emd.shape)

    # 添加哑变量及embedding, 全0向量
    concept_dim = pretrained_concept_emd.shape[1]
    concept_num = pretrained_concept_emd.shape[0] + 1
    pretrained_concept_emd = np.insert(pretrained_concept_emd, 0, np.zeros((1, concept_dim)), 0)

    relation_num = pretrained_relation_emd.shape[0] * 2 + 1
    relation_dim = pretrained_relation_emd.shape[1]
    pretrained_relation_emd = np.concatenate((pretrained_relation_emd, pretrained_relation_emd))
    pretrained_relation_emd = np.insert(pretrained_relation_emd, 0, np.zeros((1, relation_dim)), 0)

    pretrained_concept_emd = torch.FloatTensor(pretrained_concept_emd)
    pretrained_relation_emd = torch.FloatTensor(pretrained_relation_emd)

    lstm_dim = 128
    lstm_layer_num = 1
    dropout = 0.0
    bidirect = True
    batch_size = 32
    n_epochs = 8
    num_choice = 2
    qas_encoded_dim = 128
    num_random_paths = None
    graph_hidden_dim = 50
    graph_output_dim = 25
    patience = 5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dev_set = data_with_graphs_and_paths("../dataset/test.statements",
                                         "../dataset/test_path.2.pf.pruned.pnxg",
                                         "../dataset/test_path.2.pf.pruned.pickle",
                                         num_choice=2, reload=False, cut_off=3, start=0, end=None)

    # """
    train_set = data_with_graphs_and_paths("../dataset/train.statements",
                                           "../dataset/train_path.2.pf.pruned.pnxg",
                                           "../dataset/train_path.2.pf.pruned.pickle",
                                           num_choice=2, reload=False, cut_off=3, start=0, end=None)
    # """
    print("len(train_set):", len(train_set), "len(dev_set):", len(dev_set))

    model = KnowledgeAwareGraphNetworks(concept_dim, relation_dim,
                                        concept_num, relation_num, qas_encoded_dim,
                                        pretrained_concept_emd, pretrained_relation_emd,
                                        lstm_dim, lstm_layer_num, device, graph_hidden_dim, graph_output_dim,
                                        dropout=dropout, bidirect=bidirect, num_random_paths=num_random_paths,
                                        path_attention=False, qa_attention=False)
    model.to(device)

    print("checking model parameters")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("Trainable: ", name, param.size())
        else:
            print("Fixed: ", name, param.size())
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model num para#:", num_params)

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.Adam(parameters, lr=0.0005, weight_decay=0.0001, amsgrad=True)
    loss_func = torch.nn.MarginRankingLoss(margin=0.2, size_average=None, reduce=None, reduction='mean')

    no_up = 0
    best_dev_acc = 0.0
    for i in range(n_epochs):
        print('epoch: %d start!' % (i+1))
        train_epoch_kag_netowrk(train_set, batch_size, optimizer, device, model, num_choice, loss_func)

        # train_acc = eval_kag_netowrk(train_set, batch_size, device, model, num_choice)
        # print("training acc: %.5f" % train_acc, end="\t\t")

        dev_acc = eval_kag_netowrk(dev_set, batch_size, device, model, num_choice)
        print("dev acc: %.5f" % dev_acc)

        if dev_acc >= best_dev_acc:
            best_dev_acc = dev_acc
            no_up = 0
        else:
            no_up += 1
            if no_up > patience:
                break


if __name__ == "__main__":
    train()
