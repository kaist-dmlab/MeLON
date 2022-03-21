# -*- coding: UTF-8 -*-
import torch
import math
import torch.nn as nn
import logging
import numpy as np
import torch.nn.functional as F
import dgl
import copy
import argparse
import dgl.function as fn
import time
from scipy.sparse import coo_matrix
from typing import NoReturn, List

from utils import utils, autograd_hacks
from helpers.MetaReader import MetaReader


class MeLON(nn.Module):
    r"""A meta-model which provides sample-parameter adaptive learning rate.
    Wsdm consists of two components for aggregation and learning rate generation.
    Phase 1: GNN-based user-item aggregation
        1. For each target user/item, draw history of user/item
        2. Aggregate via GNN to make latent vector
        3. Concatenate user/item vector and summarize via MLP

    Phase 2: Sample-parameter adaptive learning rate generation
        1. Receive latent vector, parameter, gradient, loss
        2. For each module, output learning rate or new parameter

    Require:
        Phase 1.
            1. Target user/item pair
            2. User/item history
        Phase 2.
            1. Loss, gradient(preprocessed), parameter, previous parameter
    """

    @staticmethod
    def parse_meta_args(parser):
        parser.add_argument('--meta_model_path', type=str, default='',
                            help='Model save path.')
        parser.add_argument('--grad_clip', type=float, default=0.25,
                            help='gradient clipping')
        parser.add_argument('--max_edge', type=int, default=10000,
                            help='num_emb_edges')
        try:
            parser.add_argument('--meta_emb_size', type=int, default=20,
                                help='Size of embedding vectors.')
            parser.add_argument('--dropout', type=float, default=0,
                                help='Dropout probability for each deep layer')
        except argparse.ArgumentError as e:
            print(e)
            print("argument already registered")
            pass

        return parser

    @staticmethod
    def init_weights(m):
        if 'Linear' in str(type(m)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif 'Embedding' in str(type(m)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)

    """
    Auxiliary methods
    """
    def save_model(self, model_path=None) -> NoReturn:
        if model_path is None:
            model_path = self.model_path
        utils.check_dir(model_path)
        torch.save({'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                    model_path)
        logging.info('Save model to ' + model_path[:50] + '...')


    def save_best_model(self, model_path=None) -> NoReturn:
        if model_path is None:
            model_path = self.model_path
        utils.check_dir(model_path)
        torch.save({'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                    model_path + '_best')
        logging.info('Save model to ' + model_path[:50] + '...')

    def load_model(self, model_path=None) -> NoReturn:
        if model_path is None:
            model_path = self.model_path
        check_point = torch.load(model_path)
        self.load_state_dict(check_point['model_state_dict'])
        self.optimizer.load_state_dict(check_point['optimizer_state_dict'])
        logging.info('Load model from ' + model_path)

    def count_variables(self) -> int:
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_parameters

    def actions_before_train(self):
        pass

    def actions_before_copy(self):
        pass

    def actions_after_copy(self, model=None):
        pass


    def customize_parameters(self) -> list:
        # customize optimizer settings for different parameters
        weight_p, bias_p = [], []
        for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        optimize_dict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]
        return optimize_dict


    def __init__(self, args, corpus: MetaReader, model):
        super(MeLON, self).__init__()

        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        '''Phase 1: GAT/GIN/GCN... unify parameters, just change names! deliever two vectors'''
        self.gnn = MetaGAT(args)

        self.linear1 = nn.Linear(args.emb_size*2, args.meta_emb_size) # user/item vec -> latent vec
        self.bn1 = nn.BatchNorm1d(num_features=args.meta_emb_size)
        self.linear2 = nn.Linear(args.meta_emb_size, args.meta_emb_size) # latent vec -> latent vec
        self.bn2 = nn.BatchNorm1d(num_features=args.meta_emb_size)
        self.model_name = args.model_name

        '''Phase 2: Meta Optimizer'''
        self.metaOpt = MetaOpt(args)

        self.meta_emb_size = args.meta_emb_size
        self.optimizer = None
        self.model_path = args.meta_model_path

        self.model_clone = copy.deepcopy(model)
        self.model_clone.to(model._device)

        self.total_parameters = self.count_variables()
        logging.info('#meta params: %d' % self.total_parameters)


    def add_new_edges(self, g, u, i) -> NoReturn:
        r'''Update the affinity matrix using recent user-item interactions

        Args:
            g (dgl.heterograph)
            u (torch.Tensor of size [batch_size])
            i (torch.Tensor of size [batch_size])
        '''
        g.add_edges(u, i, etype='u_to_i')
        g.add_edges(i, u, etype='i_to_u')


    def disconnect_embedding(self, g):
        g.nodes['item'].data['emb'] = None
        g.nodes['user'].data['emb'] = None

    def connect_embedding(self, g, model):
        if self.model_name == 'BPR':
            g.nodes['user'].data['emb'] = model.u_embeddings._parameters['weight'].detach()
            g.nodes['item'].data['emb'] = model.i_embeddings._parameters['weight'].detach()

        elif self.model_name == 'GMF':
            g.nodes['user'].data['emb'] = model.mf_u_embeddings._parameters['weight'].detach()
            g.nodes['item'].data['emb'] = model.mf_i_embeddings._parameters['weight'].detach()

        elif self.model_name == 'MLP':
            g.nodes['user'].data['emb'] = model.mlp_u_embeddings._parameters['weight'].detach()
            g.nodes['item'].data['emb'] = model.mlp_i_embeddings._parameters['weight'].detach()

        elif (self.model_name == 'NCF') or (self.model_name == 'NeuMF') or (self.model_name == 'NeuGAT'):
            mf_u = model.mf_u_embeddings._parameters['weight'].detach()
            mf_i = model.mf_i_embeddings._parameters['weight'].detach()

            mlp_u = model.mlp_u_embeddings._parameters['weight'].detach()
            mlp_i = model.mlp_i_embeddings._parameters['weight'].detach()

            g.nodes['user'].data['emb'] = torch.cat((mf_u, mlp_u), dim=1)
            g.nodes['item'].data['emb'] = torch.cat((mf_i, mlp_i), dim=1)

        else:
            raise ValueError("Wrong Model Name")

        '''
        for name, params in model.named_parameters():
            if 'u_emb' in name:
                g.nodes['user'].data['emb'] = params.detach()
            elif 'i_emb' in name:
                g.nodes['item'].data['emb'] = params.detach()
        '''

    def forward(self, g, u, i, loss, model) -> torch.Tensor:
        r'''Perform user/item target/history information aggregation via GAT (Phase 1).

        Args:
            g (dgl.heterograph)
            user_id (torch.Tensor of size [batch_size])
            item_id (torch.Tensor of size [batch_size, 1+neg_items])
            loss (torch.Tensor of size [batch_size,])
            model (nn.Module): target recommender model
        '''

        #model_ = copy.deepcopy(model).to(model._device)

        u = u.unsqueeze(-1).repeat((1, i.shape[1]))  # [batch_size, 1+neg_items]
        u = u.reshape(-1) # [batch_size*(1+neg_items),]
        i = i.reshape(-1) # [batch_size*(1+neg_items),]


        assert u.shape == i.shape, "Different number of user/item"
        assert u.dim() == i.dim() == 1, "Different dimension"

        self.connect_embedding(g, model)

        # [batch_size*(1+neg_items), emb_size*2]
        ui_vec = self.gnn(g, u, i)

        latent_vec = torch.relu(self.bn1(self.linear1(ui_vec))) # [batch_size*(1+neg_items), meta_emb_size]
        latent_vec = torch.relu(self.bn2(self.linear2(latent_vec))) # [batch_size*(1+neg_items), meta_emb_size]

        """Perform parameter-sample adaptive update for selected parameters (Phase 2).

        Materials we can use in the form of: y = x*w
            x: m.activations
            y_grad/n: m.weight.grad1, or m.backprops_list[0] # needs *= m.activations.shape[0]
            w: m._parameters['weight']

        Process:
            1. Get backpropagation
            2. Modify gradient
            3. Update
            4. Average

        Note:
            1. Loss is averaged in m.backprops_list[0], multiply by number of samples
            2. Using m._parameters['weight'] can change params, not m.weight
        """

        for m, m_ in zip(model.modules(), self.model_clone.modules()):
            if isinstance(m, nn.Embedding): # Non-shared parameters
                new_params = self.metaOpt(latent_vec,
                                          m._parameters['weight'][m.weight.grad_idx].view(len(latent_vec), -1),
                                          m.weight.grad1.view(len(latent_vec), -1),
                                          loss)

                # groupby mean of result
                avg_params, indices = autograd_hacks.groupby_mean(new_params, m.weight.grad_idx.reshape(-1))

                # partially assigning params doesn't work
                tmp_params = m_._parameters['weight'].detach().clone()
                del m_._parameters['weight']
                tmp_params[indices] = avg_params
                m_._parameters['weight'] = tmp_params

            elif isinstance(m, nn.Linear): # Shared parameters
                # [in, out] -> [in*out]
                old_params = m._parameters['weight'].data.detach().view(-1)
                n_params = old_params.size(0)

                # [batch_size, meta_emb_size]
                # [batch_size, in*out]
                # [batch_size, in*out]
                new_params = self.metaOpt(latent_vec,
                                          old_params.unsqueeze_(0).expand(len(u), n_params).view(len(latent_vec), -1),
                                          m.weight.grad1.view(len(latent_vec), -1),
                                          loss)

                # mean of new parameters
                # [batch_size, in*out] -> [in*out]
                avg_params = new_params.mean(dim=0)

                # partially assigning params doesn't work
                del m_._parameters['weight']

                # [in*out] -> [in, out]
                m_._parameters['weight'] = avg_params.view_as(m._parameters['weight'])

                try:
                    # [in] -> [batch_size, in]
                    if m._parameters['bias'] is not None:
                        old_params = m._parameters['bias'].data.detach()
                        n_params = old_params.size(0)

                        new_params = self.metaOpt(latent_vec, # [batch_size, meta_emb_size]
                                                  old_params.unsqueeze_(0).expand(len(loss), n_params), # [batch_size, in]
                                                  m.bias.grad1, # [batch_size, in]
                                                  loss) # [batch_size,]

                        # mean of new parameters
                        # [batch_size, in] -> [in]
                        avg_params = new_params.mean(dim=0)

                        # partially assigning params doesn't work
                        del m_._parameters['bias']

                        # [batch_size, in] -> [in]
                        m_._parameters['bias'] = avg_params.view_as(m._parameters['bias'])
                except KeyError: pass


        #return self.model_clone


    def meta_train(self, model, meta_model, last, current, args, data):
        meta_loss = meta_model.train_meta_model(model, last, current, args, meta_model, data.g)
        loss, prediction = meta_model.train_recommender(model, current, args, meta_model, data.g)
        if data.phase == 'test':
            meta_model.add_new_edges(data.g, current['user_id'], current['item_id'][:,0])

        return model, meta_model, loss, meta_loss, prediction


    def train_meta_model(self, model, last, current, args, meta_model, g):
        r"""Train Meta-Model"""
        autograd_hacks.add_hooks(model)
        autograd_hacks.clear_backprops(model)
        prediction = model(last['user_id'], last['item_id'][:, :2])
        loss = model.loss(prediction)
        # nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        loss.backward()
        autograd_hacks.compute_grad1(model)
        meta_model(g, last['user_id'], last['item_id'][:, :2], loss, model)

        autograd_hacks.clear_backprops(model)
        #autograd_hacks.clear_backprops(model_clone)
        autograd_hacks.remove_hooks(model)
        #autograd_hacks.remove_hooks(model_clone)

        meta_model.optimizer.zero_grad()
        clone_pred = self.model_clone(current['user_id'], current['item_id'][:, :2])
        loss = self.model_clone.loss(clone_pred)
        loss.backward()
        #nn.utils.clip_grad_norm_(meta_model.parameters(), args.grad_clip)
        meta_model.optimizer.step()
        #self.copy_params(model, model_clone)

        '''
        model.load_state_dict(self.model_clone.state_dict())

        with torch.no_grad(): # prediction for evaluation
            clone_pred = model(current['user_id'], current['item_id'])
        '''

        return loss.detach().cpu().data.numpy()

    def train_recommender(self, model, current, args, meta_model, g):
        autograd_hacks.add_hooks(model)
        autograd_hacks.clear_backprops(model)
        with torch.no_grad(): # prediction for evaluation
            pred_eval = model(current['user_id'], current['item_id'])
        prediction = model(current['user_id'], current['item_id'][:, :2])
        loss = model.loss(prediction)
        # nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        loss.backward()
        autograd_hacks.compute_grad1(model)
        meta_model(g, current['user_id'], current['item_id'][:, :2], loss, model)

        autograd_hacks.clear_backprops(model)
        #autograd_hacks.clear_backprops(model_clone)
        autograd_hacks.remove_hooks(model)
        #autograd_hacks.remove_hooks(model_clone)

        model.load_state_dict(self.model_clone.state_dict())


        return loss.detach().cpu().data.numpy(), pred_eval.cpu().data.numpy()


    def leafify_params(self):
        for m in self.model_clone.modules():
            if isinstance(m, nn.Embedding) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                m._parameters['weight'] = m._parameters['weight'].detach()
                try:
                    if m._parameters['bias'] is not None:
                        m._parameters['bias'] = m._parameters['bias'].detach()
                except KeyError: pass


    @staticmethod
    def copy_params(target, source):
        '''Copy parameter to target model, gradient will not flow back to source model.
        Can't update source model with target model.
        '''
        for target, source in zip(target.parameters(), source.parameters()):
            target.data.copy_(source.data)


    @staticmethod
    def transfer_params(target, source):
        '''Source parameters will be transferred to target model.
        target model can't update itself, as it owns non-leaf parameter(source model's parameter)
        Inference with target model, and update with source model's optimizer.
        '''
        for target, source in zip(target.parameters(), source.parameters()):
            if isinstance(target, nn.Embedding) or isinstance(target, nn.Linear):
                target._parameters['weight'] = source._parameters['weight'].clone()
                try:
                    target._parameters['bias'] = source._parameters['bias'].clone()
                except KeyError: pass


class MetaGAT(nn.Module):
    def __init__(self, args):
        super(MetaGAT, self).__init__()
        if args.model_name in ['NeuMF', 'NCF', 'NeuGAT']:
            n_emb = 2
        else:
            n_emb = 1
        self.attn_fc_u = nn.Linear(2*args.emb_size*n_emb, 1, bias=False)
        self.attn_fc_i = nn.Linear(2*args.emb_size*n_emb, 1, bias=False)
        self.fc_u = nn.Linear(2*args.emb_size, args.emb_size, bias=False)
        self.fc_i = nn.Linear(2*args.emb_size, args.emb_size, bias=False)
        self.self_u = nn.Linear(args.emb_size*n_emb, args.emb_size)
        self.self_i = nn.Linear(args.emb_size*n_emb, args.emb_size)
        self.neighbor_u = nn.Linear(args.emb_size*n_emb, args.emb_size) # [n_users, emb_size]
        self.neighbor_i = nn.Linear(args.emb_size*n_emb, args.emb_size)

        self.reset_parameters()
        self.max_edge = args.max_edge


    def reset_parameters(self):
        r"""Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('leaky_relu')
        #nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc_u.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc_i.weight, gain=gain)
        #gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_u.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_i.weight, gain=gain)
        nn.init.xavier_normal_(self.self_u.weight, gain=gain)
        nn.init.xavier_normal_(self.self_i.weight, gain=gain)
        nn.init.xavier_normal_(self.neighbor_u.weight, gain=gain)
        nn.init.xavier_normal_(self.neighbor_i.weight, gain=gain)


    def forward(self, g, u, i):
        r'''Perform GAT on user/item and output vector

        Phase 1: GNN-based user-item aggregation
            0. Construct graph
            1. For each target user/item, draw history of user/item
            2. Aggregate via GNN to make latent vector
            3. Concatenate user/item vector and summarize via MLP

        Return: concatenated user/item vector
        '''

        assert len(u) == len(i), "different number of user and item"


        # Maybe change emb into z via fc introducing additional parameters.
        # equation (3) & (4) for user_hist items -> user
        start = time.time()
        affinity_u = dgl.sampling.sample_neighbors(g, {'user': torch.unique(u)}, self.max_edge)
        affinity_u['i_to_u'].apply_edges(self.edge_attention_u)
        affinity_u['i_to_u'].update_all(self.message_func, self.reduce_func, etype='i_to_u')
        u_neighbor = affinity_u.dstnodes['user'].data['h'][u] # [n_users, emb_size*n_emb]
        u_neighbor = torch.relu(self.neighbor_u(u_neighbor)) # [n_users, emb_size]
        u_self = torch.relu(self.self_u(affinity_u.nodes['user'].data['emb'][u])) # [n_users, emb_size]
        u_vec = torch.relu(self.fc_u(torch.cat((u_self, u_neighbor), dim=1)))# [n_users, emb_size]

        affinity_i = dgl.sampling.sample_neighbors(g, {'item': torch.unique(i)}, self.max_edge)
        affinity_i['u_to_i'].apply_edges(self.edge_attention_i)
        affinity_i['u_to_i'].update_all(self.message_func, self.reduce_func, etype='u_to_i')
        i_neighbor = affinity_i.dstnodes['item'].data['h'][i] # [n_items, emb_size]
        i_neighbor = torch.relu(self.neighbor_i(i_neighbor)) # [n_users, emb_size]
        i_self = torch.relu(self.self_i(affinity_i.nodes['item'].data['emb'][i])) # [n_items, emb_size]
        i_vec = torch.relu(self.fc_i(torch.cat((i_self, i_neighbor), dim=1)))# [n_items, emb_size]

        return torch.cat((u_vec, i_vec), dim=1) # [n_user=n_items, emb_size*2]


    def edge_attention_u(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['emb'], edges.dst['emb']], dim=1)
        a = self.attn_fc_u(z2)
        return {'e': F.leaky_relu(a)}

    def edge_attention_i(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['emb'], edges.dst['emb']], dim=1)
        a = self.attn_fc_i(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['emb'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

class MetaLSTMCell2(nn.Module):
    """C_t = f_t * C_{t-1} + i_t * \tilde{C_t}"""
    def __init__(self, input_size, emb_size):
        super(MetaLSTMCell2, self).__init__()
        """Args:
            input_size (int): cell input size, default = 20
            emb_size (int): should be 1
            n_learner_params (int): number of learner's parameters
        """
        self.input_size = input_size
        self.emb_size = emb_size
        self.WF = nn.Parameter(torch.Tensor(input_size, emb_size))
        self.WI = nn.Parameter(torch.Tensor(input_size, emb_size))
        self.bI = nn.Parameter(torch.Tensor(1, emb_size))
        self.bF = nn.Parameter(torch.Tensor(1, emb_size))

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            nn.init.uniform_(weight, -0.01, 0.01)

        # want initial forget value to be high and input value to be low so that
        # model starts with gradient descent
        nn.init.uniform_(self.bF, 4, 6)
        nn.init.uniform_(self.bI, -5, -4)

    def init_cI(self, flat_params):
        pass

    def forward(self, inputs, latent_vec, params, grad):
        """Args:
            inputs (torch.Tensor of size [batch_size, params, input_size]): outputs from previous LSTM
            latent_vec (torch.Tensor of size [batch_size, meta_emb_size]): vector
            params (torch.Tensor of size [batch_size, params]): outputs from previous LSTM
            grad (torch.Tensor of size [batch_size, params]): gradients from learner

            Possible use them after debugging...
            hx = [f_prev, i_prev, c_prev]:
                f (torch.Tensor of size [n_learner_params, 1]): forget gate
                i (torch.Tensor of size [n_learner_params, 1]): input gate
                c (torch.Tensor of size [n_learner_params, 1]): flattened learner parameters
        """
        n_batch, n_param, _ = inputs.size()

        c_prev = params.unsqueeze(-1)

        # [batch_size, params, meta_emb]
        #latent_vec = latent_vec.unsqueeze(1).expand_as(inputs)
        latent_vec = latent_vec.unsqueeze(1).expand(n_batch, n_param, -1)

        # self.WF = nn.Parameter(torch.Tensor(input_size + 2, emb_size))
        # f_t = sigmoid(W_f * [grad_t, loss_t, theta_{t-1}, f_{t-1}] + b_f)
        # [batch_size, params, 1]
        f_next = torch.matmul(torch.cat((inputs, latent_vec), -1), self.WF) + self.bF.expand_as(c_prev)
        # i_t = sigmoid(W_i * [grad_t, loss_t, theta_{t-1}, i_{t-1}] + b_i)
        i_next = torch.matmul(torch.cat((inputs, latent_vec), -1), self.WI) + self.bI.expand_as(c_prev)
        # next cell/params
        c_next = torch.sigmoid(f_next).mul(c_prev) - torch.sigmoid(i_next).mul(grad.unsqueeze(-1))

        return c_next.squeeze_(-1) #, [f_next, i_next, c_next]

    def extra_repr(self):
        s = '{input_size}, {emb_size}'
        return s.format(**self.__dict__)


class MetaOpt(nn.Module):
    def __init__(self, args):
        super(MetaOpt, self).__init__()
        self.input_size = 4
        self.emb_size = 20
        self.meta_emb_size = args.meta_emb_size
        self.linear = nn.Linear(self.input_size, self.emb_size)
        self.metalstm = MetaLSTMCell2(input_size=self.emb_size+self.meta_emb_size, emb_size=1)


    def forward(self, latent_vec, params, grad, loss):
        """Args:
            latent_vec (torch.Tensor of [batch_size, meta_emb_size])
            params (torch.Tensor of size [batch_size, param_size])
            grad (torch.Tensor of size [batch_size, param_size])
            loss (torch.Tensor of size [batch_size,])
        """

        loss_prep = loss.data.unsqueeze(-1).expand_as(grad) # [batch_size, params]
        grad_prep = utils.preprocess_grad_loss(grad).requires_grad_(False) # [batch_size, params, 2]
        loss_prep = utils.preprocess_grad_loss(loss_prep).requires_grad_(False) # [batch_size, params, 2]

        inputs = torch.cat((loss_prep, grad_prep), -1)   # [batch_size, params, 4]

        del grad_prep, loss_prep

        hx = self.linear(inputs)
        new_params = self.metalstm(hx, latent_vec, params, grad)

        return new_params
