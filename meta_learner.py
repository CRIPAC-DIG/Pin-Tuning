import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import numpy as np
from models import MAML, PinTuning, NCESoftmaxLoss
from dataset import FewshotMolDataset, dataset_sampler
from sklearn.metrics import roc_auc_score

import logging
from tqdm import tqdm
from copy import deepcopy

logger = logging.getLogger()


class MovingAVG:
    def __init__(self):
        self.count = 0
        self.avg = 0

    def get_avg(self):
        return self.avg

    def update(self, x):
        self.count += 1
        self.avg = self.avg + (x - self.avg) / self.count


class MetaLearner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if int(args.gpu) >= 0 else 'cpu')
        self.dataset = FewshotMolDataset(root=args.data_root, name=args.dataset)
        self.train_task_range, self.test_task_range = self.dataset.train_task_range, self.dataset.test_task_range
        model = PinTuning(task_num=self.dataset.total_tasks,
                        train_task_num=self.dataset.n_task_train,
                        args=args).to(self.device)
        self.maml = MAML(model, lr=args.inner_lr, first_order=not args.second_order, anil=False, allow_unused=True)
        self.opt = optim.AdamW(self.maml.parameters(), lr=args.meta_lr, weight_decay=args.weight_decay)
        self.cls_criterion = nn.BCEWithLogitsLoss()

        self.n_support, self.n_query = args.n_support, args.n_query
        self.inner_update_step = args.inner_update_step

        if self.args.train_auxi_task_num is None:
            self.train_auxi_task_num = len(self.train_task_range) - 1
        else:
            self.train_auxi_task_num = min(args.train_auxi_task_num, len(self.train_task_range) - 1)
        if self.args.test_auxi_task_num is None:
            self.test_auxi_task_num = len(self.train_task_range)
        else:
            self.test_auxi_task_num = min(args.test_auxi_task_num, len(self.train_task_range))

        self.nce_loss = NCESoftmaxLoss(t=args.nce_t)
        self.args.pool_num = min(self.args.pool_num, len(self.train_task_range))

    def update_inner(self, s_data, q_data, task_id, auxi_tasks):
        sampled_task = torch.tensor([task_id] + auxi_tasks).to(self.device)
        s_y, q_y = s_data.y[:, sampled_task], q_data.y[:, sampled_task]
        model = self.maml.clone()
        model.train()

        original_embeddings = []
        original_embeddings.append(model.module.mol_encoder.gnn.x_embedding1.weight.clone())
        original_embeddings.append(model.module.mol_encoder.gnn.x_embedding2.weight.clone())
        for gconv in model.module.mol_encoder.gnn.gnns:
            original_embeddings.append(gconv.edge_embedding1.weight.clone())
            original_embeddings.append(gconv.edge_embedding2.weight.clone())
        original_embedding = torch.cat(original_embeddings, dim=0)

        for _ in range(self.args.inner_update_step):
            s_logit, q_logit, s_label, q_label, graph_f = model(s_data, q_data, s_y, q_y, sampled_task)
            inner_loss = self.cls_criterion(s_logit, s_label)
            model.adapt(inner_loss, allow_nograd=True)
        s_logit, q_logit, s_label, q_label, graph_f = model(s_data, q_data, s_y, q_y, sampled_task)
        eval_loss = self.cls_criterion(q_logit, q_label)

        updated_embeddings = []
        updated_embeddings.append(model.module.mol_encoder.gnn.x_embedding1.weight.clone())
        updated_embeddings.append(model.module.mol_encoder.gnn.x_embedding2.weight.clone())
        for gconv in model.module.mol_encoder.gnn.gnns:
            updated_embeddings.append(gconv.edge_embedding1.weight.clone())
            updated_embeddings.append(gconv.edge_embedding2.weight.clone())
        updated_embedding = torch.cat(updated_embeddings, dim=0)

        # the Emb-BWC loss is equivalent to the $L_2$ penalty when H is approximated by the identity matrix
        emb_bwc_loss = torch.norm(original_embedding - updated_embedding, p='fro')

        return eval_loss, graph_f, emb_bwc_loss

    def train_step(self, epoch):
        selected_ids, selected_tasks = self.sample_tasks(epoch)

        eval_losses = []
        emb_bwc_losses = []
        graph_f1s, graph_f2s = [], []
        for task_id, (s_data1, q_data1, s_data2, q_data2) in zip(selected_ids, selected_tasks):
            auxi_tasks = self.sample_auxiliary(task_id, self.train_task_range, self.train_auxi_task_num)
            eval_loss1, graph_f1, emb_bwc_loss1 = self.update_inner(s_data1, q_data1, task_id, auxi_tasks)
            eval_loss2, graph_f2, emb_bwc_loss2 = self.update_inner(s_data2, q_data2, task_id, auxi_tasks)
            eval_losses += [eval_loss1, eval_loss2]
            emb_bwc_losses += [emb_bwc_loss1, emb_bwc_loss2]
            graph_f1s.append(graph_f1)
            graph_f2s.append(graph_f2)

        tgt_f1, tgt_f2 = torch.vstack(graph_f1s), torch.vstack(graph_f2s)
        loss_contr = self.nce_loss(tgt_f1, tgt_f2)

        loss_cls = torch.stack(eval_losses).mean()

        loss_emb_bwc_mean = torch.stack(emb_bwc_losses).mean()
        
        self.opt.zero_grad()
        loss = loss_cls + loss_contr * self.args.contr_w

        if self.args.use_norm_loss:
            loss += loss_emb_bwc_mean * self.args.norm_w

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.maml.parameters(), 1)
        self.opt.step()

        return loss_cls.item()

    def test_step(self, test_auxi_task_num=None):
        auc_scores = []
        for task_i in tqdm(self.test_task_range, desc='eval'):
            s_data, q_data = dataset_sampler(self.dataset, self.n_support, self.n_query,
                                                tgt_id=task_i, inductive=True)
            s_data = Batch.from_data_list(s_data).to(self.device)
            test_auxi_task_num = self.test_auxi_task_num if test_auxi_task_num is None else test_auxi_task_num
            auxi_tasks = self.sample_auxiliary(task_i, self.train_task_range, test_auxi_task_num)
            sampled_task = torch.tensor([task_i] + auxi_tasks).to(self.device)
            s_y = s_data.y[:, sampled_task]
            model = self.maml.clone()
            model.train()
            # inner update
            adapt_q_iter = iter(DataLoader(q_data, batch_size=self.args.n_query, shuffle=True))
            for _ in range(self.args.inner_update_step):
                adapt_q_data = next(adapt_q_iter)
                adapt_q_data = adapt_q_data.to(self.device)
                adapt_q_y = adapt_q_data.y[:, sampled_task]
                s_logit, q_logit, s_label, q_label, _, = model(s_data, adapt_q_data,
                                                                s_y, adapt_q_y, sampled_task)
                inner_loss = self.cls_criterion(s_logit, s_label)
                model.adapt(inner_loss, allow_nograd=True)
            model.eval()

            y_pred, y_true = [], []
            query_loader = DataLoader(q_data, batch_size=self.args.test_batch_size, num_workers=2, shuffle=False)
            with torch.no_grad():
                for iter_q_data in query_loader:
                    iter_q_data = iter_q_data.to(self.device)
                    iter_q_y = iter_q_data.y[:, sampled_task]
                    _, q_logit, _, q_label, _, = model(s_data, iter_q_data, s_y, iter_q_y, sampled_task)
                    q_logit = torch.sigmoid(q_logit).cpu().view(-1)
                    q_label = q_label.cpu().view(-1)
                    y_pred.append(q_logit)
                    y_true.append(q_label)

                y_true = torch.cat(y_true, dim=0).numpy()
                y_pred = torch.cat(y_pred, dim=0).numpy()
                score = roc_auc_score(y_true, y_pred)
                auc_scores.append(score)
        return np.mean(auc_scores)

    def sample_tasks(self, epoch):
        def sample_data(tgt_id):
            s_data1, q_data1 = dataset_sampler(self.dataset, self.n_support, self.n_query, tgt_id)
            s_data1 = Batch.from_data_list(s_data1).to(self.device)
            q_data1 = Batch.from_data_list(q_data1).to(self.device)
            s_data2, q_data2 = dataset_sampler(self.dataset, self.n_support, self.n_query, tgt_id)
            s_data2 = Batch.from_data_list(s_data2).to(self.device)
            q_data2 = Batch.from_data_list(q_data2).to(self.device)
            return s_data1, q_data1, s_data2, q_data2


        tasks_pool = []
        tasks_pool_ids = np.random.choice(self.train_task_range, self.args.pool_num, replace=False)
        for task_id in tasks_pool_ids:
            s_data1, q_data1, s_data2, q_data2 = sample_data(task_id)
            tasks_pool.append((s_data1, q_data1, s_data2, q_data2))
        return tasks_pool_ids, tasks_pool

    def sample_auxiliary(self, tgt_task_id, auxi_task_range, auxi_task_num):
        if tgt_task_id in auxi_task_range:
            auxi_task_range = deepcopy(auxi_task_range)
            auxi_task_range.remove(tgt_task_id)

        selected_ids = np.random.choice(auxi_task_range, auxi_task_num, replace=False).tolist()
        return selected_ids