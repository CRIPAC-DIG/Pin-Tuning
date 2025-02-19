from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.data import Batch, Data

from .graph_connector import InductiveGraphConnector


class EdgeUpdateNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer=3, top_k=5, batch_norm=False, dropout=.0):
        super(EdgeUpdateNetwork, self).__init__()
        self.top_k = top_k

        num_dims_list = [hidden_dim] * n_layer
        if n_layer > 1:
            num_dims_list[0] = 2 * hidden_dim
        if n_layer > 3:
            num_dims_list[1] = 2 * hidden_dim

        # layers
        layer_list = OrderedDict()
        for l in range(len(num_dims_list)):
            # set layer
            layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=num_dims_list[l - 1] if l > 0 else in_dim,
                                                       out_channels=num_dims_list[l],
                                                       kernel_size=1,
                                                       bias=False)  # in: [N, C_in, H, W]
            if batch_norm:
                layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=num_dims_list[l], )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if dropout > 0:
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=dropout)

        layer_list['conv_out'] = nn.Conv2d(in_channels=num_dims_list[-1],
                                           out_channels=1,
                                           kernel_size=1)
        self.sim_network = nn.Sequential(layer_list)

    def forward(self, node_feat):
        """
        :param node_feat: [bs, num_node, d] [36, 300]
        :return:
        """
        x_i = node_feat.unsqueeze(1)
        x_j = node_feat.unsqueeze(0)
        x_ij = torch.abs(x_i - x_j)
        x_ij = torch.transpose(x_ij, 0, 2)
        x_ij = torch.exp(-x_ij)
        x_ij = x_ij.unsqueeze(0)
        sim_val = self.sim_network(x_ij).squeeze()
        diag_mask = 1.0 - torch.eye(node_feat.size(0)).to(node_feat.device)
        adj_val = torch.sigmoid(sim_val) * diag_mask
        if self.top_k >= 0:
            n1, n2 = adj_val.size()
            k = min(self.top_k, n1)
            topk, indices = torch.topk(adj_val, k)
            mask = torch.zeros_like(adj_val)
            mask = mask.scatter(1, indices, 1)
            adj_val = adj_val * mask
        return adj_val


class NodeUpdateNetwork(MessagePassing):
    def __init__(self, in_emb_dim, out_emb_dim, num_bond_type=4, norm=False, batch_norm=False, edge_type=True,
                 dropout=0., aggr='mean'):
        super(NodeUpdateNetwork, self).__init__(aggr=aggr)
        self.edge_type = edge_type
        self.neigh_linear = nn.Linear(in_emb_dim, out_emb_dim)
        self.root_linear = nn.Linear(in_emb_dim, out_emb_dim)
        self.edge_emb = nn.Embedding(num_bond_type, out_emb_dim)
        nn.init.xavier_uniform_(self.edge_emb.weight.data)
        self.norm = norm
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_emb_dim)
        else:
            self.batch_norm = None
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU()

    def forward(self, x, edge_index, edge_attr, edge_weight):
        """
        :param x: [num_node, d]
        :param edge_index: [2, num_edge]
        :param edge_attr: [num_edge, num_attr]
        :param edge_weight: [num_edge, 1]
        :return:
        """
        edge_embeddings = self.edge_emb(edge_attr[:, 0])

        neigh_x = self.neigh_linear(x)
        msg = self.propagate(edge_index, x=neigh_x, edge_attr=edge_embeddings, edge_weight=edge_weight)
        msg += self.root_linear(x)
        if self.norm:
            msg = F.normalize(msg, p=2, dim=-1)
        if self.batch_norm:
            msg = self.batch_norm(msg)
        return self.dropout(self.act(msg))

    def message(self, x_j, edge_attr, edge_weight):
        if self.edge_type:
            return (x_j + edge_attr) * edge_weight
        else:
            return x_j * edge_weight


class Context_Encoder(nn.Module):
    def __init__(self,
                 in_dim,
                 num_layer,
                 edge_hidden_dim,
                 edge_n_layer,
                 total_tasks,
                 train_tasks,
                 batch_norm=False,
                 edge_type=True,
                 top_k=-1,
                 dropout=0.,
                 pre_dropout=0.,
                 nan_w=0.,
                 nan_type='nan'):
        super(Context_Encoder, self).__init__()
        self.dropout = dropout
        self.total_tasks = total_tasks
        self.num_layer = num_layer
        self.nan_w = nan_w
        self.nan_type = nan_type
        if pre_dropout > 0:
            self.pre_dropout = nn.Dropout(pre_dropout)
        else:
            self.pre_dropout = None
        self.task_emb = nn.Embedding(total_tasks, in_dim)
        self.task_emb.weight.data[train_tasks:, :] = 0
        for i in range(num_layer):
            module_edge = EdgeUpdateNetwork(in_dim=in_dim, hidden_dim=edge_hidden_dim, n_layer=edge_n_layer,
                                            top_k=top_k, batch_norm=batch_norm, dropout=dropout)
            module_node = NodeUpdateNetwork(in_emb_dim=in_dim, out_emb_dim=in_dim,
                                            dropout=dropout, batch_norm=batch_norm, edge_type=edge_type)
            self.add_module(f'node_layer{i}', module_node)
            self.add_module(f'edge_layer{i}', module_edge)
        self.graph_pooling = nn.Sequential(nn.Linear(in_dim, in_dim//2), nn.ReLU(),
                                            nn.Linear(in_dim//2, in_dim))

    def get_task_emb(self, task_id):
        return self.task_emb(task_id)

    def forward(self):
        pass

    def forward_inductive(self, sample_emb, task_id, support_y, query_y):
        """
        :param sample_emb: [n_q, n_s+1, d]
        :param task_id: [n_task]
        :param support_y: [n_s, n_task]
        :param query_y: [n_q, n_task]
        :param return_auxi
        :return: support_logit: [n_q, n_s]
                 query_logit: [n_q, 1]
                 tgt_s_y: [n_q, n_s]
                 tgt_q_y: [n_q, 1]
                 graph_emb: [d]

        :param sample_emb: [16, 21, 300]
        :param task_id: [9]
        :param support_y: [20, 1]
        :param query_y: [16, 1]
        :return: support_logit: [16, 20]
                 query_logit: [16, 1]
                 tgt_s_y: [16, 20]
                 tgt_q_y: [16, 1]
                 graph_emb: [300]
        """
        task_emb = self.task_emb(task_id)
        n_task, n_q, n_s = support_y.shape[1], query_y.shape[0], support_y.shape[0]
        if self.pre_dropout:
            sample_emb = self.pre_dropout(sample_emb)
            task_emb = self.pre_dropout(task_emb)

        tgt_s_y, tgt_q_y, tgt_s_idx, tgt_q_idx, edge_ls, edge_type_ls, edge_w_ls = \
            InductiveGraphConnector.connect_task_and_sample(support_y, query_y, self.nan_w, self.nan_type)
        input_emb = torch.cat([task_emb, sample_emb], dim=0)

        for i in range(self.num_layer):
            sample_adj = self._modules['edge_layer{}'.format(i)](sample_emb)
            edges, edge_types, edge_ws = InductiveGraphConnector.connect_graph(sample_adj, edge_ls, edge_type_ls,
                                                                                edge_w_ls, n_task)

            data = Data(x=input_emb, edge_index=edges, edge_type=edge_types, edge_w=edge_ws)

            input_emb = self._modules['node_layer{}'.format(i)](data.x,
                                                                data.edge_index,
                                                                data.edge_type,
                                                                data.edge_w)
            input_emb = input_emb.contiguous()
            task_emb, sample_emb = input_emb[:len(task_id), :], input_emb[len(task_id):, :]

        support_context = torch.cat([input_emb[tgt_s_idx[:, 0], :],
                                    input_emb[tgt_s_idx[:, 1], :]], dim=-1)
        query_context = torch.cat([input_emb[tgt_q_idx[:, 0], :],
                                  input_emb[tgt_q_idx[:, 1], :]], dim=-1)
        
        task_context = input_emb[0, :]
        support_mol_context = input_emb[tgt_s_idx[:, 0], :]
        query_mol_context = input_emb[tgt_q_idx[:, 0], :]
                
        tgt_s_y = tgt_s_y.contiguous()
        graph_emb = input_emb[0] + torch.sigmoid(input_emb[n_task:n_task+n_s].mean(0))

        return tgt_s_y, tgt_q_y, graph_emb, task_context, support_mol_context, query_mol_context, support_context, query_context
