# -*- coding: UTF-8 -*-

import torch
import logging
import os
import numpy as np
import copy
import pickle
import time
from tqdm import tqdm
from random import randint, choice
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset
from torch.nn.utils.rnn import pad_sequence
from typing import NoReturn, List

from utils import utils
from helpers.MetaReader import MetaReader


class MetaModel(torch.nn.Module):
    reader = 'MetaReader'
    runner = 'MetaRunner'
    extra_log_args = []

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--model_path', type=str, default='',
                            help='Model save path.')
        parser.add_argument('--num_neg', type=int, default=99,
                            help='The number of negative items for testing. 1 for training.')
        parser.add_argument('--dropout', type=float, default=0,
                            help='Dropout probability for each deep layer')
        parser.add_argument('--buffer', type=int, default=1,
                            help='Whether to buffer feed dicts for dev/test')
        return parser

    @staticmethod
    def init_weights(m):
        if 'Linear' in str(type(m)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif 'Embedding' in str(type(m)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def __init__(self, args, corpus: MetaReader):
        super(MetaModel, self).__init__()
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_path = args.model_path
        self.num_neg = args.num_neg
        self.dropout = args.dropout
        self.buffer = args.buffer
        self.g = copy.deepcopy(corpus.g)
        self.item_num = corpus.n_items
        self.optimizer = None
        self.check_list = list()  # observe tensors in check_list every check_epoch

        self._define_params()
        self.total_parameters = self.count_variables()
        logging.info('#params: %d' % self.total_parameters)

    """
    Methods must to override
    """
    def _define_params(self) -> NoReturn:
        self.item_bias = torch.nn.Embedding(self.item_num, 1)

    def forward(self, feed_dict: dict) -> torch.Tensor:
        """
        :param feed_dict: batch prepared in Dataset
        :return: prediction with shape [batch_size, n_candidates]
        """
        i_ids = feed_dict['item_id']
        prediction = self.item_bias(i_ids)
        return prediction.view(feed_dict['batch_size'], -1)

    """
    Methods optional to override
    """
    '''
    def loss(self, predictions: torch.Tensor, reduction:str = 'mean') -> torch.Tensor:
        """
        MSE Loss: should do sigmoid() 
        """
        criterion = torch.nn.MSELoss(reduction='none')
        pos_pred, neg_pred = predictions[:, 0:1], predictions[:, 1:2] # 1 pos : 1 neg
        pos_loss = criterion(pos_pred, torch.ones_like(pos_pred).to(pos_pred.device))
        neg_loss = criterion(neg_pred, torch.zeros_like(neg_pred).to(neg_pred.device))
        """
        print("pos_pred:", pos_pred.view(-1)[:5])
        print("neg_pred:", neg_pred.view(-1)[:5])
        print("pos_loss:", pos_loss.view(-1)[:5])
        print("neg_loss:", neg_loss.view(-1)[:5])
        """
        loss = torch.add(pos_loss, neg_loss)
        if reduction == 'mean':
            loss = loss.mean()
        return loss
    '''


    def loss(self, predictions: torch.Tensor, reduction:str = 'mean') -> torch.Tensor:
        """
        BPR ranking loss with optimization on multiple negative samples
        @{Recurrent neural networks with top-k gains for session-based recommendations}
        :param predictions: [batch_size, -1], the first column for positive, the rest for negative
        :return:
        """
        pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:2] # 1 pos : 1 neg
        neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
        loss = -((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1).log()#.mean()
        # neg_pred = (neg_pred * neg_softmax).sum(dim=1)
        # loss = F.softplus(-(pos_pred - neg_pred)).mean()
        # ↑ For numerical stability, we use 'softplus(-x)' instead of '-log_sigmoid(x)'
        if reduction == 'mean':
            loss = loss.mean()
        return loss



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

    def actions_before_train(self):  # e.g. re-initial some special parameters
        pass

    def actions_after_train(self):  # e.g. save selected parameters
        pass

    """
    Define dataset class for the model
    """

    class Dataset(BaseDataset):
        def __init__(self, model, args, corpus, phase: str):
            self.model = model  # model object reference
            self.corpus = corpus  # reader object reference
            self.phase = phase
            self.neg_items = None # if phase == 'train' else self.data['neg_items']
            # ↑ Sample negative items before each epoch during training
            self.train_ratio = args.train_ratio
            self.mini_batch_path = corpus.mini_batch_path
            self.graph_path = corpus.graph_path
            self.batch_size = args.batch_size
            self.buffer = self.model.buffer and self.phase != 'train'
            if phase == 'train':
                self.n_batches = int(corpus.n_batches*args.train_ratio)
                logging.info("train n_batches: %s" %str(self.n_batches))
            else:
                self.n_batches = corpus.n_batches-int(corpus.n_batches*args.train_ratio)
                self._prepare_neg_items()
                logging.info("test n_batches: %s" %str(self.n_batches))
                assert corpus.n_test == len(self.neg_items), "Neg items not equal"


        def _prepare_neg_items(self):
            r"""Sample and prepare negative items for test data.
            Warning: check number of items before usage"""
            logging.info('Prepare neg_items')
            start = time.time()
            num_neg = self.model.num_neg
            neg_items = torch.zeros(size=(self.corpus.n_test, num_neg), dtype=torch.int64)
            for i, u in enumerate(self.corpus.user_list[self.corpus.n_train:]):
                user_clicked_set = copy.deepcopy(self.corpus.user_clicked_set[u])
                # By copying, it may not collide with other process with same user index
                for j in range(num_neg):
                    neg_item = self._randint_w_exclude(user_clicked_set)
                    neg_items[i][j] = neg_item
                    user_clicked_set = np.append(user_clicked_set, neg_item)

            self.neg_items = neg_items
            logging.info('Complete neg_item preparation: %s seconds' %str(int(time.time()-start)))

        def __len__(self):
            '''
            Returns the number of batches
            '''
            if self.phase == 'train':
                return self.n_batches
            else:
                return self.n_batches


        def __getitem__(self, index: int) -> dict:
            last = self._get_feed_dict(index)
            current = self._get_feed_dict(index, current=True)
            return last, current


        def _get_feed_dict(self, index: int, current=False) -> dict:
            r"""Return user-item mini-batch and index/value adjacency-batch (torch.FloatTensor).

            Process:
                train
                    last [batch_size*2, 2]
                        randomly sample positive/negative user/item among history
                        requires user_hist_set, item_hist_set
                        use current batch to index user/item

                    current [batch_size, 2]
                        positive: read from saved mini-batch
                        negative: randomly pick using self._sample_items

                test
                    index + n_batches*train_ratio
                    last [batch_size*2, 2]
                        let's prepare everything needed here
                        positive: dictionary based. if not, use current add user
                        negative: randomly sample negative item
                        use current batch to index user/item
                        requires preliminary dictionary

                    current [batch_size, 2]
                        positive: read from saved mini-batch
                        negative: already prepared negative item using self._prepare_neg_items

            Input:
                index: index of the batch

            """

            if self.phase == 'test':
                index += int(self.corpus.n_batches*self.train_ratio)

            if current:  # return [batch_size, -1]
                user_id, item_id = torch.load(os.path.join(self.mini_batch_path, str(index)+'.pt')).T
                neg_items = self._sample_neg_items(index*self.batch_size,
                                                  index*self.batch_size+len(user_id))
                item_id = torch.cat((item_id.reshape(-1, 1), neg_items), axis=-1)
                feed_dict = {'user_id': user_id, #(batch_size, )
                             'item_id': item_id} #(batch_size, 1+neg_items)

                return feed_dict

            else: # return [batch_size*2, 2]
                if self.phase == 'train': # randomly sample negative items
                    user_id, item_id = torch.load(os.path.join(self.mini_batch_path, str(index)+'.pt')).T
                    # Exception handling: If a user has no previous interactions, then the current item is selected

                    # same user, different item: (u, i'), (u, -i')
                    # for positive: get user, and pick another used item
                    # for negative: get user, and pick random negative item
                    pos_items_u = torch.zeros(size=(len(user_id), 1), dtype=torch.int64)
                    neg_items_u = torch.zeros(size=(len(user_id), 1), dtype=torch.int64)
                    for idx, user in enumerate(user_id):
                        user_hist_set = copy.deepcopy(self.corpus.user_hist_set[user.item()])
                        user_clicked_set = copy.deepcopy(self.corpus.user_clicked_set[user.item()])
                        pos_items_u[idx] = choice(user_hist_set)
                        neg_items_u[idx] = self._randint_w_exclude(user_clicked_set)
                    items_u = torch.cat((pos_items_u, neg_items_u), axis=-1)

                    # different user, same item: (u', i), (u', -i)
                    # for positive: get another user, and pick item
                    # for negative: get another user, and pick random negative item
                    user_id_ = torch.zeros_like(user_id)
                    for idx, item in enumerate(item_id): # pick u'
                        user_id_[idx] = choice(self.corpus.item_hist_set[item.item()])

                    neg_items_u_ = torch.zeros(size=(len(user_id), 1), dtype=torch.int64)
                    for idx, user in enumerate(user_id_):
                        user_clicked_set = copy.deepcopy(self.corpus.user_clicked_set[user.item()])
                        neg_items_u_[idx] = self._randint_w_exclude(user_clicked_set)
                    items_u_ = torch.cat((item_id.reshape(-1, 1), neg_items_u_), axis=-1)

                    user_id = torch.cat((user_id, user_id_), axis=0)
                    item_id = torch.cat((items_u, items_u_), axis=0)
                    feed_dict = {'user_id': user_id, # [batch_size*2,]
                                 'item_id': item_id} # [batch_size*2, 2]

                    return feed_dict

                else: # read saved positive-negative pairs. Prepared in helpers/MetaReader.py
                    feed_dict = torch.load(os.path.join(self.corpus.last_batch_path, str(index)+'.pt'))
                    return feed_dict


        def _sample_neg_items(self, index, index_end):
            r"""Sample positive (for meta_batch) and negative items (for meta and current batch).
            For training: randomly sample 1 item.
            For testing: randomly sample 99 item.
            GIL for same user_clicked_set will be avoided by copying it
            """
            if self.phase == 'train': num_neg = 1
            elif self.phase == 'test':
                num_neg = self.model.num_neg
                if self.buffer:
                    if self.neg_items is None:
                        logging.info('neg_items not prepared, check code')
                        self._prepare_neg_items()
                    _index = index - self.corpus.n_train
                    _index_end = index_end - self.corpus.n_train
                    return self.neg_items[_index:_index_end]
            neg_items = torch.zeros(size=(self.batch_size, num_neg), dtype=torch.int64)
            for idx, user in enumerate(self.corpus.user_list[index:index_end]): # Automatic coverage?
                user_clicked_set = copy.deepcopy(self.corpus.user_clicked_set[user])
                # By copying, it may not collide with other process with same user index
                for neg in range(num_neg):
                    neg_item = self._randint_w_exclude(user_clicked_set)
                    neg_items[idx][neg] = neg_item
                    # Skip below: one neg for train
                    # user_clicked_set = np.append(user_clicked_set, neg_item)

            return neg_items

        def _randint_w_exclude(self, clicked_set):
            randItem = randint(1, self.corpus.n_items-1)
            return self._randint_w_exclude(clicked_set) if randItem in clicked_set else randItem
