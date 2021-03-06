# -*- coding: UTF-8 -*-

import os
import gc
import copy
import torch
import logging
import numpy as np
import pandas as pd

from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List, NoReturn
from collections import defaultdict

from utils import utils, autograd_hacks
from models.MetaModel import MetaModel


class MetaRunner(object):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--epoch', type=int, default=100,
                            help='Number of epochs.')
        parser.add_argument('--check_epoch', type=int, default=1,
                            help='Check some tensors every check_epoch.')
        parser.add_argument('--early_stop', type=int, default=5,
                            help='The number of epochs when dev results drop continuously.')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='Learning rate.')
        parser.add_argument('--l2', type=float, default=0,
                            help='Weight decay in optimizer.')
        parser.add_argument('--batch_size', type=int, default=256,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size', type=int, default=256,
                            help='Batch size during testing.')
        parser.add_argument('--optimizer', type=str, default='Adam',
                            help='optimizer: GD, Adam, Adagrad, Adadelta')
        parser.add_argument('--num_workers', type=int, default=4,
                            help='Number of processors when prepare batches in DataLoader')
        parser.add_argument('--pin_memory', type=int, default=1,
                            help='pin_memory in DataLoader')
        parser.add_argument('--topk', type=str, default='[1,5,10,20,50]',
                            help='The number of items recommended to each user.')
        parser.add_argument('--metric', type=str, default='["NDCG","HR"]',
                            help='metrics: NDCG, HR')
        return parser

    @staticmethod
    def evaluate_method(predictions: np.ndarray, topk: list, metrics: list) -> Dict[str, float]:
        """
        :param predictions: (-1, n_candidates) shape, the first column is the score for ground-truth item
        :param topk: top-K values list
        :param metrics: metrics string list
        :return: a result dict, the keys are metrics@topk
        """
        evaluations = dict()
        sort_idx = (-predictions).argsort(axis=1)
        gt_rank = np.argwhere(sort_idx == 0)[:, 1] + 1
        for k in topk:
            hit = (gt_rank <= k)
            for metric in metrics:
                key = '{}@{}'.format(metric, k)
                if metric == 'HR':
                    evaluations[key] = hit.mean(dtype=np.float16)
                elif metric == 'NDCG':
                    evaluations[key] = (hit / np.log2(gt_rank + 1)).mean(dtype=np.float16)
                else:
                    raise ValueError('Undefined evaluation metric: {}.'.format(metric))
        return evaluations

    def __init__(self, args):
        self.epoch = args.epoch
        self.check_epoch = args.check_epoch
        self.early_stop = args.early_stop
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.keys = args.keys
        self.l2 = args.l2
        self.optimizer_name = args.optimizer
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.topk = eval(args.topk)
        self.metrics = [m.strip().upper() for m in eval(args.metric)]
        self.result_file = args.result_file
        self.meta_weighting = args.meta_weighting
        self.meta_name = args.meta_name
        self.main_metric = '{}@{}'.format(self.metrics[0], self.topk[0])  # early stop based on main_metric

        self.time = None  # will store [start_time, last_step_time]

    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def _build_optimizer(self, model):
        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == 'gd':
            logging.info("Optimizer: GD")
            optimizer = torch.optim.SGD(
                model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
        elif optimizer_name == 'adagrad':
            logging.info("Optimizer: Adagrad")
            optimizer = torch.optim.Adagrad(
                model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
        elif optimizer_name == 'adadelta':
            logging.info("Optimizer: Adadelta")
            optimizer = torch.optim.Adadelta(
                model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
        elif optimizer_name == 'adam':
            logging.info("Optimizer: Adam")
            optimizer = torch.optim.Adam(
                model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
        else:
            raise ValueError("Unknown Optimizer: " + self.optimizer_name)
        return optimizer

    def eval_termination(self, criterion: List[float]) -> bool:
        if len(criterion) > 20 and utils.non_increasing(criterion[-self.early_stop:]):
            return True
        elif len(criterion) - criterion.index(max(criterion)) > 20:
            return True
        return False

    def evaluate(self, model: torch.nn.Module, data: MetaModel.Dataset, topks: list, metrics: list) -> Dict[str, float]:
        """
        Evaluate the results for an input dataset.
        :return: result dict (key: metric@k)
        """
        predictions = self.predict(model, data)
        return self.evaluate_method(predictions, topks, self.metrics)

    def predict(self, model: torch.nn.Module, data: MetaModel.Dataset) -> np.ndarray:
        """
        The returned prediction is a 2D-array, each row corresponds to all the candidates,
        and the ground-truth item poses the first.
        Example: ground-truth items: [1, 2], 2 negative items for each instance: [[3,4], [5,6]]
                 predictions order: [[1,3,4], [2,5,6]]
        """
        model.eval()
        predictions = list()
        dl = DataLoader(data, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
                        pin_memory=self.pin_memory)
                        #collate_fn=data.collate_batch, pin_memory=self.pin_memory)
        for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
            batch['batch_size'] = len(batch['user_id'])
            prediction = model(utils.batch_to_gpu(utils.batch_to_gpu(batch), model._device))
            predictions.extend(prediction.cpu().data.numpy())
        return np.array(predictions)


    ########################## methods for MeLON ###########################

    def print_res(self,
                  model: torch.nn.Module,
                  data_dict: Dict[str, MetaModel.Dataset],
                  args,
                  meta_model) -> str:
        """
        Construct the final result string before/after training
        :return: test result string
        """
        _, _, result_dict_list = self.fit_online(model, data_dict['test'], args, meta_model)
        res_str_first = '(' + utils.format_metric(result_dict_list[0]) + ')'
        res_str_last = '(' + utils.format_metric(result_dict_list[-1]) + ')'
        return ' '.join((res_str_first, res_str_last))

    def train(self,
              model: torch.nn.Module,
              data_dict: Dict[str, MetaModel.Dataset],
              args,
              meta_model=None,
              online_ratio=0.5) -> NoReturn:

        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)

        if self.meta_weighting and meta_model.optimizer is None:
            meta_model.optimizer = self._build_optimizer(meta_model)

        main_metric_results, test_results = list(), list()
        self._check_time(start=True)

        # train offline for designated epochs
        try:
            for epoch in range(args.last_epoch, self.epoch):
                # Fit
                if epoch > 0:
                    model.load_model()
                    if args.meta_weighting > 0:
                        meta_model.load_model()

                self._check_time()
                loss, meta_loss = self.fit_offline(model, data_dict['train'],
                                                   args, meta_model, epoch=epoch + 1)
                training_time = self._check_time()


                # Observe selected tensors
                if len(model.check_list) > 0 and self.check_epoch > 0 and epoch % self.check_epoch == 0:
                    utils.check(model.check_list)

                """
                # Do when not enough GPU
                model.to('cpu')
                if args.meta_weighting > 0:
                    meta_model.to('cpu')
                """

                model_ = copy.deepcopy(model)
                if args.meta_name in ['MeLON']:
                    meta_model.leafify_params()
                meta_model_ = copy.deepcopy(meta_model)

                data_dict['test'].g = copy.deepcopy(data_dict['train'].g)

                loss_eval, flag, tmp_result = self.fit_online(model_,
                                                        data_dict['test'],
                                                        args, meta_model_)

                test_results.append(tmp_result) # test_results (epoch, test_batch_length, keys[metric])

                testing_time = self._check_time()

                main_metric_results.append(test_results[-1][0][self.main_metric])

                model.save_model()
                if args.meta_weighting >0:
                    meta_model.save_model()

                # Print first and last loss/test
                logging.info("Epoch {:<3} loss={:<.4f} meta_loss={:<.4f} loss_eval={:<.4f} [{:<.1f} s] {}=({:<.4f}) {}=({:<.4f}) [{:<.1f} s] ".format(
                             epoch + 1, loss, meta_loss, loss_eval, training_time, 'first_test ' + self.main_metric,
                             test_results[-1][0][self.main_metric],
                             'last_test  ' + self.main_metric, test_results[-1][-1][self.main_metric],
                             testing_time))

                if flag:
                    pd.DataFrame(test_results).to_csv(self.result_file + '.csv', index=False)
                    logging.info("Nonzero prediction continues, early stop training")
                    logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
                    exit(1)

                pd.DataFrame(tmp_result).to_csv(self.result_file+'/'+str(epoch)+'.csv', index=False)


                # Save model and early stop
                if max(main_metric_results) == main_metric_results[-1] or \
                        (hasattr(model, 'stage') and model.stage == 1):
                    model.save_best_model()

        except KeyboardInterrupt:
            logging.info("Early stop manually")
            exit_here = input("Exit completely without evaluation? (y/n) (default n):")
            if exit_here.lower().startswith('y'):
                logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
                exit(1)

        # Find the best dev result across iterations
        best_epoch = main_metric_results.index(max(main_metric_results))

        logging.info(os.linesep + "Best Iter={:>5}\t test_batch 0=({}) test_batch {}=({}) [{:<.1f} s] ".format(
                     best_epoch+1,
                     test_results[best_epoch][0],
                     len(test_results[0])-1, test_results[best_epoch][-1],
                     self.time[1] - self.time[0]))

        pd.DataFrame(test_results).to_csv(self.result_file + '.csv', index=False)


    def fit_offline(self,
                    model: torch.nn.Module,
                    data: MetaModel.Dataset,
                    args,
                    meta_model,
                    epoch=-1) -> float:
        r'''Run offline evaluation: randomly select mini-batch and train'''

        gc.collect()
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            model.to(model._device)

        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)

        if self.meta_weighting and meta_model.optimizer is None:
            meta_model.optimizer = self._build_optimizer(meta_model)




        loss_lst, meta_loss_lst = list(), list()
        dl = DataLoader(data, batch_size=1, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)
        for last, current in tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
            current = utils.batch_to_gpu(utils.squeeze_dict(current), model._device)
            current['batch_size'] = len(current['user_id'])
            if self.meta_weighting: # MeLON
                last = utils.batch_to_gpu(utils.squeeze_dict(last), model._device)
                last['batch_size'] = len(last['user_id'])

                model, meta_model, loss, meta_loss, prediction = meta_model.meta_train(model,
                                                                                  meta_model,
                                                                                  last,
                                                                                  current,
                                                                                  args,
                                                                                  data)
                meta_loss_lst.append(meta_loss)
            else: # default
                loss, _ = self.train_recommender_vanilla(model, current)

            loss_lst.append(loss)

        data.neg_items = None
        return np.mean(loss_lst).item(), np.mean(meta_loss_lst).item()


    def fit_online(self,
                model: torch.nn.Module,
                #data_dict: Dict[str, MetaModel.Dataset],
                data: MetaModel.Dataset,
                args,
                meta_model,
                epoch=-1) -> float:
        r'''Run prequential evaluation'''

        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        if torch.cuda.is_available():
            model.to(model._device)
            if self.meta_weighting:
                meta_model.to(model._device)

        model.num_neg = args.num_neg

        gc.collect()
        torch.cuda.empty_cache()

        loss_lst, meta_loss_lst, eval_result = list(), list(), list()
        dl = DataLoader(data, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
        for last, current in tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
            current = utils.batch_to_gpu(utils.squeeze_dict(current), model._device)
            current['batch_size'] = len(current['user_id'])

            if self.meta_weighting: # MeLON
                last = utils.batch_to_gpu(utils.squeeze_dict(last), model._device)
                last['batch_size'] = len(last['user_id'])

                model, meta_model, loss, meta_loss, prediction = meta_model.meta_train(model,
                                                                                      meta_model,
                                                                                      last,
                                                                                      current,
                                                                                      args,
                                                                                      data)
            else: # default
                loss, prediction = self.train_recommender_vanilla(model, current)
            loss_lst.append(loss)

            tmp_eval_result_dict = self.evaluate_method(prediction, self.topk, self.metrics)

            eval_result.append(tmp_eval_result_dict) # (test_batch_length, keys[metric])
        flag = len(np.nonzero(prediction)[0]) == 0 or np.isinf(prediction).any() or np.isnan(prediction).any()

        # return nan for the meta_loss_lst, if not meta_weighting mode
        return np.mean(loss_lst).item(), flag,  eval_result


    def train_recommender_vanilla(self, model, current):
        r"""Train recommender system"""
        # Train recommender
        model.train()

        # Get recommender's prediction and loss from the ``current'' data at t
        prediction = model(current['user_id'], current['item_id'])
        loss = model.loss(prediction, reduction='mean')

        # Update the recommender
        model.optimizer.zero_grad()
        loss.backward()

        model.optimizer.step()

        return loss.cpu().data.numpy(), prediction.cpu().data.numpy()

