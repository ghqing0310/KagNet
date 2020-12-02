import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import dgl.function as fn
import numpy as np
import random
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h': h}


class GraphConvLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GraphConvLayer, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


class GCN_Encoder(nn.Module):
    def __init__(self, concept_dim, hidden_dim, output_dim, pretrained_concept_emd, concept_emd=None):
        super(GCN_Encoder, self).__init__()

        self.gcn1 = GraphConvLayer(concept_dim, hidden_dim, F.relu)
        self.gcn2 = GraphConvLayer(hidden_dim, output_dim, F.relu)

        if pretrained_concept_emd is not None and concept_emd is None:
            self.concept_emd = nn.Embedding(num_embeddings=pretrained_concept_emd.size(0),
                                        embedding_dim=pretrained_concept_emd.size(1))
            self.concept_emd.weight = nn.Parameter(pretrained_concept_emd)  # init
        elif pretrained_concept_emd is None and concept_emd is not None:
            self.concept_emd = concept_emd

    def forward(self, g):
        features = self.concept_emd(g.ndata["cncpt_ids"])
        print("features",  features.shape)
        x = self.gcn1(g, features)
        print(x.shape)
        x = self.gcn2(g, x)
        print(x.shape)
        g.ndata['h'] = x
        return g


class KnowledgeAwareGraphNetworks(nn.Module):
    def __init__(self, concept_dim, relation_dim,
                 concept_num, relation_num,
                 pretrained_concept_emd, pretrained_relation_emd,
                 device, graph_hidden_dim, graph_output_dim, num_random_paths=None
                 ):

        super(KnowledgeAwareGraphNetworks, self).__init__()
        self.num_random_paths = num_random_paths
        self.concept_dim = concept_dim
        self.relation_dim = relation_dim

        self.concept_emd = nn.Embedding(concept_dim, concept_num)
        self.relation_emd = nn.Embedding(relation_num, relation_dim)
        self.graph_hidden_dim = graph_hidden_dim
        self.graph_output_dim = graph_output_dim

        # random init the embeddings
        self.relation_emd.weight = nn.Parameter(pretrained_relation_emd)
        self.concept_emd.weight = nn.Parameter(pretrained_concept_emd)

        self.nonlinear = nn.LeakyReLU()

        self.device = device
        self.hidden2output = nn.Sequential(
            nn.Linear(self.graph_output_dim, 1),  # binary classification
            nn.Sigmoid()
         )

        self.hidden2output.apply(weight_init)

        self.graph_encoder = GCN_Encoder(self.concept_dim, self.graph_hidden_dim, self.graph_output_dim,
                                         pretrained_concept_emd=None, concept_emd=self.concept_emd)

    # qas_vec is the concat of the question concept, answer concept, and the statement
    def forward(self, graphs, concept_mapping_dicts):
        self.device = self.concept_emd.weight.device  # multiple GPUs need to specify device
        final_vecs = []

        output_graphs = self.graph_encoder(graphs)
        output_concept_embeds = torch.cat((output_graphs.ndata["h"], torch.zeros(1, self.graph_output_dim).to(self.device))) # len(output_concept_embeds) as padding
        print(output_concept_embeds.shape)
        new_concept_embed = torch.tensor((output_graphs.ndata["h"])).to(self.device)
        print(new_concept_embed.shape)
        final_vecs.append(new_concept_embed)
        logits = self.hidden2output(torch.stack(final_vecs))
        return logits