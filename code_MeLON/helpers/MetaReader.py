# -*- coding: UTF-8 -*-

import argparse
import os
import time
import pickle
import json
import logging
import math
import copy
import torch
import sys
import dgl
from random import randint, choice

import scipy.sparse
import pandas as pd
import numpy as np

from utils import utils
from torch.nn.utils.rnn import pad_sequence
from torch import nn


class MetaReader(object):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--path', type=str, default='../data/',
                            help='Input data dir.')
        parser.add_argument('--suffix', type=str, default='MeLON',
                            help='Input data dir of MeLON.')
        parser.add_argument('--dataset', type=str, default='Grocery_and_Gourmet_Food',
                            help='Choose a dataset.')
        parser.add_argument('--sep', type=str, default='\t',
                            help='Sep of csv file.')
        parser.add_argument('--train_ratio', type=float, default=0.8,
                            help='Ratio of the train dataset')
        parser.add_argument('--duplicate', type=int, default=1,
                            help='Coalesce duplicate elements in adjacency matrix')
        parser.add_argument('--fname', type=str, default='freq',
                            help='Freq (> 20 records) or whole')
        return parser


    def __init__(self, args):
        self.sep = args.sep
        self.prefix = args.path
        self.suffix = args.suffix
        self.dataset = args.dataset
        self.train_ratio = args.train_ratio
        self.batch_size = args.batch_size
        self.fname = args.fname
        self.random_seed = args.random_seed

        t0 = time.time()
        self._read_data()
        print(self.data_df['user_id'].max(),len(self.data_df['user_id'].unique()))
        print(self.data_df['item_id'].max(),len(self.data_df['item_id'].unique()))

        logging.info('Counting dataset statistics...')
        self.n_users, self.n_items = self.data_df['user_id'].max()+1, self.data_df['item_id'].max()+1
        self.dataset_size = len(self.data_df)
        self.n_batches = math.ceil(self.dataset_size/self.batch_size)
        logging.info('"# user": {}, "# item": {}, "# entry": {}'.format(self.n_users-1, self.n_items-1, self.dataset_size))

        path = os.path.join(self.prefix, self.dataset, self.suffix, self.fname)
        if not os.path.exists(path):
            os.mkdir(path)
        del path

        logging.info('Saving data into mini-batch pickle files')
        self.user_list = self.data_df['user_id'].to_numpy()

        self._save_user_clicked_set()
        self._save_hist_set()
        self._save_mini_batch()
        self._save_last_batch()
        self._save_heterograph()

        del self.data_df

        logging.info('Done! [{:<.2f} s]'.format(time.time() - t0) + os.linesep)


    def _read_data(self):
        logging.info('Reading data from \"{}\", dataset = \"{}\", suffix = \"{}\", fname = \"{}\" '.format(self.prefix, self.dataset, self.suffix, self.fname))
        df = pd.read_csv(os.path.join(self.prefix, self.dataset, self.suffix, self.fname +'.csv'), sep=self.sep)  # Let the main runner decide the ratio of train/test
        self.data_df = df.loc[:, ['user_id', 'item_id']]#.values.astype(np.int64) # (number of items, 2)


    def _save_user_clicked_set(self):
        user_clicked_set_path = os.path.join(self.prefix, self.dataset, self.suffix, self.fname, 'user_clicked_set.txt')
        logging.info('Load user_clicked_set')

        try:
            self.user_clicked_set = pickle.load(open(user_clicked_set_path, 'rb'))
            logging.info("Successfully loaded saved user_clicked_set")
        except FileNotFoundError as e:
            logging.info('File not found, create user_clicked_set')
            self.user_clicked_set = self.data_df.groupby(['user_id'])['item_id'].unique().to_dict()
            pickle.dump(self.user_clicked_set, open(user_clicked_set_path, 'wb'))
            logging.info('Saved user_clicked_set')


    def _save_hist_set(self):
        '''Prepare previous interactions of users and items during offline meta-model training.
        '''
        item_hist_set_path = os.path.join(self.prefix, self.dataset, self.suffix, self.fname, 'item_hist_set.txt')
        user_hist_set_path = os.path.join(self.prefix, self.dataset, self.suffix, self.fname, 'user_hist_set.txt')
        logging.info('Load hist_set')

        try:
            self.item_hist_set = pickle.load(open(item_hist_set_path, 'rb'))
            self.user_hist_set = pickle.load(open(user_hist_set_path, 'rb'))
            logging.info("Successfully loaded saved hist_set")
        except FileNotFoundError as e:
            logging.info('File not found, create hist_set')
            train_df = self.data_df.loc[:int(self.n_batches*self.train_ratio)*self.batch_size,:]
            self.item_hist_set = train_df.groupby(['item_id'])['user_id'].unique().to_dict()
            self.user_hist_set = train_df.groupby(['user_id'])['item_id'].unique().to_dict()
            pickle.dump(self.item_hist_set, open(item_hist_set_path, 'wb'))
            pickle.dump(self.user_hist_set, open(user_hist_set_path, 'wb'))
            logging.info('Saved hist_set')


    def _save_mini_batch(self):
        self.mini_batch_path = os.path.join(self.prefix, self.dataset, self.suffix, self.fname, 'mini_batch')
        if not os.path.exists(self.mini_batch_path):
            os.mkdir(self.mini_batch_path)

        for batch_idx in range(self.n_batches):
            ui_batch = torch.from_numpy(self.data_df[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size].values.astype(np.int64)) # (batch_size, 2)
            torch.save(ui_batch, open(os.path.join(self.mini_batch_path, str(batch_idx)+'.pt'), 'wb'))


    def _save_last_batch(self):
        '''Save previous interaction of user and item in current mini-batch for online testing.
        Given user-item interaction (u, i) in current mini-batch, Prepare previous interaction of user and item

        The resulting user-item interactions (torch.LongTensor size of [batch_size*2, 2]) are saved in last_batch_path

            # different user, same item: (u', i), (u', -i)
            # for positive: get user, and pick another used item
            # for negative: get user, and pick random negative item

            # same user, different item: (u, i'), (u, -i')
            # for positive: get user, and pick another used item
            # for negative: get user, and pick random negative item

        Exception handling:
            If a user has no previous interactions, then the current item is selected.
            If an item has no previous interactions, then the current user is selected.
            -> Randomly sample previous user/item
        Prepare (torch.LongTensor size of [batch_size*2, 2])
        '''

        self.last_batch_path = os.path.join(self.prefix, self.dataset, self.suffix, self.fname, 'last_batch')
        if not os.path.exists(self.last_batch_path):
            os.mkdir(self.last_batch_path)

        test_start_batch_idx = int(self.n_batches*self.train_ratio)
        train_df = self.data_df.loc[:test_start_batch_idx*self.batch_size,:]

        ui_last = {u:None for u in range(self.n_users)}
        ui_last_ = train_df.groupby(['user_id'])['item_id'].unique().apply(lambda x: x[-1]).to_dict()
        ui_last.update(ui_last_)

        iu_last = {i:None for i in range(self.n_items)}
        iu_last_ = train_df.groupby(['item_id'])['user_id'].unique().apply(lambda x: x[-1]).to_dict()
        iu_last.update(iu_last_)

        for batch_idx in range(test_start_batch_idx, self.n_batches):
            user_id, item_id = torch.from_numpy(self.data_df[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size].values.astype(np.int64)).T # (batch_size, 2)

            # same user, different item: (u, i'), (u, -i')
            # for positive: get user, and pick user's last item
            # for negative: get user, and pick random negative item
            pos_items_u = torch.zeros(size=(len(user_id), 1), dtype=torch.int64)
            neg_items_u = torch.zeros(size=(len(user_id), 1), dtype=torch.int64)
            for idx, user in enumerate(user_id):
                user_clicked_set = copy.deepcopy(self.user_clicked_set[user.item()])
                last_item = ui_last[user.item()]
                pos_items_u[idx] = last_item if last_item is not None else item_id[idx]
                neg_items_u[idx] = self._randint_w_exclude(user_clicked_set)
            items_u = torch.cat((pos_items_u, neg_items_u), axis=-1)

            # different user, same item: (u', i), (u', -i)
            # for positive: get item's last user, and pick item
            # for negative: get item's last user, and pick random negative item
            user_id_ = torch.zeros_like(user_id)
            for idx, item in enumerate(item_id): # pick u'
                last_user = iu_last[item.item()]
                user_id_[idx] = last_user if last_user is not None else user_id[idx]

            neg_items_u_ = torch.zeros(size=(len(user_id), 1), dtype=torch.int64)
            for idx, user in enumerate(user_id_):
                user_clicked_set = copy.deepcopy(self.user_clicked_set[user.item()])
                neg_items_u_[idx] = self._randint_w_exclude(user_clicked_set)
            items_u_ = torch.cat((item_id.reshape(-1, 1), neg_items_u_), axis=-1)

            # Update last interaction history
            for user, item in zip(user_id, item_id):
                ui_last[user.item()] = item.item()
                iu_last[item.item()] = user.item()

            # Save last interaction mini-batch
            user_id = torch.cat((user_id, user_id_), axis=0)
            item_id = torch.cat((items_u, items_u_), axis=0)

            last_batch = {'user_id': user_id, # [batch_size*2,]
                          'item_id': item_id} # [batch_size*2, 2]

            torch.save(last_batch, open(os.path.join(self.last_batch_path, str(batch_idx)+'.pt'), 'wb'))


    def _randint_w_exclude(self, clicked_set):
        randItem = randint(1, self.n_items-1)
        return self._randint_w_exclude(clicked_set) if randItem in clicked_set else randItem


    def _save_heterograph(self):
        self.graph_path = os.path.join(self.prefix, self.dataset, self.suffix, self.fname, 'heterograph_train.bin')
        logging.info('Load heterograph_train')
        try:
            g_list, _ = dgl.load_graphs(self.graph_path,[0])
            self.g = g_list[0]
            logging.info("Successfully loaded heterograph_train")
        except Exception as e:
            logging.info('File not found, create heterograph_train')
            u, i = self.data_df[:int(self.n_batches*self.train_ratio)*self.batch_size].values.T
            self.g = dgl.heterograph({('user', 'u_to_i', 'item'): (u, i),
                                      ('item', 'i_to_u', 'user'): (i, u)},
                                      num_nodes_dict={'user':self.n_users, 'item':self.n_items})
            dgl.save_graphs(self.graph_path, self.g)
            logging.info('Successfully saved heterograph_train')




    # Below are deprecated methods
    def _save_adj_batch(self):
        """Generate user/item history indices/counts (torch.LongTensor) for each batch and save them.
        Process:

            Preparation
            1. Construct offline user-item interaction matrix (scipy.sparse.csr_matrix)

            For each offline batch
            1. Index user/item for each batch and their value
            2. Pad_sequence
            3. Save them

            For each online batch
            1. Change offline matrix to LIL (scipy.sparse.lil_matrix)
            2. Increment user/item interaction
            3. Index user/item for each batch
            4. Construct batch matrix (torch.sparse.FloatTensor)
            5. Save them

        """

        # Preparation
        if self.adj is None:
            self._prepare_adj()

        # Offline
        self.adj_batch_path = os.path.join(self.prefix, self.dataset, self.suffix, self.fname, 'adj_batch')
        if not os.path.exists(self.adj_batch_path):
            os.mkdir(self.adj_batch_path)

        for batch_idx in range(int(self.n_batches*self.train_ratio)):
            users, items = self.adj_node_idx[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size].T

            self._save_hist_seq(users, batch_idx, 'u')
            self._save_hist_seq(items, batch_idx, 'i', transpose=True)

        # Online
        for batch_idx in range(int(self.n_batches*self.train_ratio), self.n_batches):
            # Add new interactions
            self.adj = self.adj.tolil()
            for user, item in self.adj_node_idx[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]:
                self.adj[user, item] += 1
            self.adj = self.adj.tocsr()

            users, items = self.adj_node_idx[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size].T

            self._save_hist_seq(users, batch_idx, 'u')
            self._save_hist_seq(items, batch_idx, 'i', transpose=True)


    def _save_hist_seq(self, batch, batch_idx, msg, transpose=False):
        indices = []
        data = []
        adj = self.adj.T if transpose else self.adj
        for i in batch:
            indices.append(torch.from_numpy(adj.getrow(i).indices))
            data.append(torch.from_numpy(adj.getrow(i).data))

        indices = pad_sequence(indices, batch_first=True, padding_value=0)
        data = pad_sequence(data, batch_first=True, padding_value=0)

        torch.save(indices, os.path.join(self.adj_batch_path, msg+'_idx'+str(batch_idx)+'.pt'))
        torch.save(data, os.path.join(self.adj_batch_path, msg+'_val'+str(batch_idx)+'.pt'))


    def update_adj(self, batch_idx):
        """Add current batch to adjacency matrix for online update"""
        if self.cur_idx == batch_idx:
            logging.info('Adjacency matrix not updated')
            return

        new_indices = torch.cat((self.adj._indices(), self.adj_node_idx[batch_idx:batch_idx+batch_size].T), axis=1)
        new_vals = torch.cat((self.adj._values(), self.adj_edge_val[batch_idx:batch_idx+batch_size]))
        self.adj = torch.sparse_coo_tensor(new_indices, new_vals)
        self.coalesce()



    def init_adj(self):
        self.cur_idx = int(self.n_batches*self.train_ratio)*self.batch_size # number of offline items
        self.adj = scipy.sparse.csr_matrix((self.adj_edge_val[:self.cur_idx],
                                            self.adj_node_idx[:self.cur_idx].T),
                                            shape=(self.n_users, self.n_items))
        logging.info('Adjacency matrix created with type: %s and shape: %s' %(str(type(self.adj)), str(self.adj.shape)))


    def _prepare_adj(self):
        logging.info('Preparing adjacency matrix')
        #self.adj_node_idx = torch.from_numpy(self.data_df.loc[:, ['user_id', 'item_id']].values).to(dtype=torch.int32) # (number of items, 2)
        #self.adj_edge_val = torch.ones(len(self.adj_node_idx), dtype=torch.int16) # (number_of_items)
        self.adj_node_idx = self.data_df.loc[:, ['user_id', 'item_id']].values.astype(np.int64) # (number of items, 2)
        self.adj_edge_val = np.ones(len(self.adj_node_idx), dtype=np.int16) # (number_of_items)

        self.init_adj()


    # Deprecated
    def _coalesce(self):
        """Coalesce duplicate edges into single (one-valued) edge.
        Require adj. matrix (torch.sparse_coo_tensor)
        """
        self.adj = self.adj.coalesce()
        assert self.adj.is_coalesced()

        if self.duplicate == 0: # No duplicate edge: edge value is always one
            self.adj = torch.sparse_coo_tensor(self.adj._indices(), self.adj._values().fill_(1)).coalesce()



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser = MetaReader.parse_data_args(parser)
    args, extras = parser.parse_known_args()

    args.path = '../../data/'
    corpus = MetaReader(args)

    corpus_path = os.path.join(args.path, args.dataset, 'mesia', 'Corpus.pkl')
    logging.info('Save corpus to {}'.format(corpus_path))
    pickle.dump(corpus, open(corpus_path, 'wb'))
