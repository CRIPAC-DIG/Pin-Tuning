from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base_encoder import GNN_Encoder_Frozen, GNN_Encoder_with_Adapter
from .relation import Context_Encoder

class PinTuning(nn.Module):
    def __init__(self, task_num, train_task_num, args):
        super(PinTuning, self).__init__()
        self.mol_encoder_frozen = GNN_Encoder_Frozen(num_layer=args.mol_num_layer,
                                       emb_dim=args.emb_dim,
                                       JK=args.JK,
                                       drop_ratio=args.mol_dropout,
                                       graph_pooling=args.mol_graph_pooling,
                                       gnn_type=args.mol_gnn_type,
                                       batch_norm=args.mol_batch_norm,
                                       load_path=args.mol_pretrain_load_path)

        self.relation_net = Context_Encoder(in_dim=args.emb_dim,
                                            num_layer=args.rel_layer,
                                            edge_n_layer=args.rel_edge_n_layer,
                                            edge_hidden_dim=args.rel_edge_hidden_dim,
                                            total_tasks=task_num,
                                            train_tasks=train_task_num,
                                            batch_norm=args.rel_batch_norm,
                                            top_k=args.rel_top_k,
                                            dropout=args.rel_dropout,
                                            pre_dropout=args.rel_pre_dropout,
                                            nan_w=args.rel_nan_w,
                                            nan_type=args.rel_nan_type,
                                            edge_type=args.rel_edge_type)
        
        self.mol_encoder = GNN_Encoder_with_Adapter(num_layer=args.mol_num_layer,
                                       emb_dim=args.emb_dim,
                                       JK=args.JK,
                                       drop_ratio=args.mol_dropout,
                                       graph_pooling=args.mol_graph_pooling,
                                       gnn_type=args.mol_gnn_type,
                                       batch_norm=args.mol_batch_norm,
                                       load_path=args.mol_pretrain_load_path,
                                       adapter_hidden_dim=args.adapter_hidden_dim,
                                       layer_norm=args.mol_layer_norm)

        self.classifier = nn.Sequential(nn.Linear(4*args.emb_dim, args.emb_dim), nn.BatchNorm1d(args.emb_dim), nn.ReLU(),
                                        nn.Linear(args.emb_dim, 1))

    def encode_mol(self, data):
        return self.mol_encoder_frozen(data.x, data.edge_index, data.edge_attr, data.batch)

    def forward(self, s_data, q_data, s_y, q_y, sampled_task):
        s_feat, q_feat = self.encode_mol(s_data), self.encode_mol(q_data)
        sample_feat = torch.cat([s_feat, q_feat], dim=0)

        s_label, q_label, graph_f, task_context, support_mol_context, query_mol_context, support_context, query_context = self.relation_net.forward_inductive(sample_feat, sampled_task, s_y, q_y)

        # ************** contextualized mol encoding **************
        contextualized_s_feat = self.mol_encoder(s_data, task_context, support_mol_context)
        contextualized_q_feat = self.mol_encoder(q_data, task_context, query_mol_context)

        # ************** compute s_logit and q_logit **************
        s_logit = self.classifier(torch.cat([s_feat, support_context, contextualized_s_feat], dim=-1))
        q_logit = self.classifier(torch.cat([q_feat, query_context, contextualized_q_feat], dim=-1))

        return s_logit, q_logit, s_label, q_label, graph_f
