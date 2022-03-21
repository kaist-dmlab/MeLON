# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn

from models.general.BPR import BPR


class NCF(BPR):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--layers', type=str, default='[64, 64, 64, 64]',
                            help="Size of each layer.")
        return BPR.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.layers = eval(args.layers)
        super().__init__(args, corpus)

    def _define_params(self):
        self.mf_u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.mf_i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.mlp_u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.mlp_i_embeddings = nn.Embedding(self.item_num, self.emb_size)

        self.mlp = nn.ModuleList([])
        pre_size = 2 * self.emb_size
        for i, layer_size in enumerate(self.layers):
            self.mlp.append(nn.Linear(pre_size, layer_size, bias=False))
            pre_size = layer_size
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.prediction = nn.Linear(pre_size + self.emb_size, 1, bias=False)

        self.u_bias = nn.Embedding(self.user_num, 1)
        self.i_bias = nn.Embedding(self.item_num, 1)


    def forward(self, u_ids, i_ids):
        self.check_list = []
        #u_ids = feed_dict['user_id']  # [batch_size]
        #i_ids = feed_dict['item_id']  # [batch_size, -1]

        u_ids = u_ids.unsqueeze(-1).repeat((1, i_ids.shape[1]))  # [batch_size, -1]

        mf_u_vectors = self.mf_u_embeddings(u_ids)
        mf_i_vectors = self.mf_i_embeddings(i_ids)
        mlp_u_vectors = self.mlp_u_embeddings(u_ids)
        mlp_i_vectors = self.mlp_i_embeddings(i_ids)

        mf_vector = mf_u_vectors * mf_i_vectors
        mlp_vector = torch.cat([mlp_u_vectors, mlp_i_vectors], dim=-1)
        for layer in self.mlp:
            mlp_vector = layer(mlp_vector).relu()
            mlp_vector = self.dropout_layer(mlp_vector)

        output_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        prediction = self.prediction(output_vector)

        user_bias = self.u_bias(u_ids).view_as(prediction)
        item_bias = self.i_bias(i_ids).view_as(prediction)

        prediction = prediction + user_bias + item_bias
        return prediction.view(len(u_ids), -1)
