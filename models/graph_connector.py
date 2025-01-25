from copy import deepcopy
import torch

nan_type_dict = {'0': 1, '1': 2, 'nan': 3}


def get_edge(label_matrix, label, shift, bi_direct=True):
    if label is None:
        # Each row in the result contains the indices of a non-zero element in input.
        # sample -> task
        edge = torch.nonzero(torch.isnan(label_matrix), as_tuple=False)
    else:
        edge = torch.nonzero(label_matrix == label, as_tuple=False)
    edge[:, 0] = edge[:, 0] + shift
    if bi_direct:
        edge_copy = deepcopy(edge)
        edge_copy[:, 0], edge_copy[:, 1] = edge[:, 1], edge[:, 0]
        edge = torch.cat([edge, edge_copy])
    return edge


class InductiveGraphConnector:
    @staticmethod
    def connect_task_and_sample(support_y, query_y, nan_w=0., nan_type='nan'):
        """
        :param support_y: [n_s, y]
        :param query_y: [n_q, y]
        :param nan_w: int
        :param nan_type
        :return: tgt_s_y: [n_s, 1]
                 tgt_q_y: [n_q, 1]
                 tgt_s_idx: [n_s, 2]
                 tgt_q_idx: [1, 2]
                 edge_ls: List of Tensor: [2, edge]
                 edge_type_ls: List of Tensor: [n_edge, 1]
                 edge_w_ls: List of Tensor: [n_edge, 1]
        """
        nan_idx = nan_type_dict[nan_type]
        tgt_s_y, tgt_q_y = support_y[:, [0]], query_y[:, [0]]
        n_task = support_y.shape[1]
        n_q, n_s = query_y.shape[0], support_y.shape[0]
        
        auxi_query_y = query_y[:, 1:]
        support_edge_0 = get_edge(support_y, 0, n_task)
        support_edge_1 = get_edge(support_y, 1, n_task)
        query_edge_0 = get_edge(auxi_query_y, 0, n_task + n_s)
        query_edge_1 = get_edge(auxi_query_y, 1, n_task + n_s)
        if nan_w == 0:
            edge = torch.cat([support_edge_0, query_edge_0,
                                support_edge_1, query_edge_1])
            edge_type = [1] * (len(support_edge_0) + len(query_edge_0)) + \
                            [2] * (len(support_edge_1) + len(query_edge_1))
            edge_type = torch.tensor(edge_type).to(edge.device)
            edge_w = torch.tensor([1.] * len(edge_type)).to(edge.device)
        else:
            support_edge_nan = get_edge(support_y, None, n_task)
            query_edge_nan = get_edge(auxi_query_y, None, n_task + n_s)
            edge = torch.cat([support_edge_0, query_edge_0,
                                support_edge_1, query_edge_1,
                                support_edge_nan, query_edge_nan])
            edge_0_n, edge_1_n, edge_nan_n = len(support_edge_0) + len(query_edge_0), \
                                                len(support_edge_1) + len(query_edge_1), \
                                                len(support_edge_nan) + len(query_edge_nan)
            edge_type = [1] * edge_0_n + [2] * edge_1_n + [nan_idx] * edge_nan_n 
            edge_type = torch.tensor(edge_type).to(edge.device)
            edge_w = torch.tensor([1.] * (edge_0_n + edge_1_n) + [nan_w] * edge_nan_n).to(edge.device)
        edge = edge.transpose(0, 1)
        edge_w = edge_w.unsqueeze(-1)
        edge_type = edge_type.unsqueeze(-1)

        tgt_s_idx = torch.tensor([list(range(n_task, n_task + n_s)), [0] * n_s]).transpose(0, 1)
        tgt_q_idx = torch.tensor([list(range(n_task + n_s, n_task + n_s + n_q)), [0] * n_q]).transpose(0, 1)
        tgt_s_idx, tgt_q_idx = tgt_s_idx.to(support_y.device), tgt_q_idx.to(support_y.device)
        return tgt_s_y, tgt_q_y, tgt_s_idx, tgt_q_idx, edge, edge_type, edge_w

    @staticmethod
    def connect_graph(adj, edge, edge_type, edge_w, n_task):
        """
        :param adj: [n_s+n_q, n_s+n_q]
        :param edge_ls: [2, n_edge]
        :param edge_type_ls: [n_edge, 1]
        :param edge_w_ls: [n_edge, 1]
        :param n_task:
        :return: edges: [2, n_edge]
                 edge_types: [n_edge, 1]
                 edge_ws: [n_edge, 1]
        """

        sample_edge = torch.nonzero(adj > 0, as_tuple=False)
        sample_w = adj[sample_edge[:, 0], sample_edge[:, 1]]
        if sample_edge.shape[0] != 0:
            sample_edge = sample_edge + n_task
            edge = torch.cat([edge, sample_edge.transpose(0, 1)], dim=1)
            edge_type = torch.cat(
                [edge_type, torch.tensor([0] * sample_edge.shape[0]).unsqueeze(1).to(adj.device)], dim=0)
            edge_w = torch.cat([edge_w, sample_w.unsqueeze(1)], dim=0)

        return edge, edge_type, edge_w
