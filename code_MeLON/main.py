# -*- coding: UTF-8 -*-

import os
import sys
import pickle
import logging
import argparse
import numpy as np
import torch
import copy

from helpers import MetaReader, MetaRunner
from models import  MetaModel, MeLON
from models.general import BPR, NCF
from utils import utils


def parse_global_args(parser):
    parser.add_argument('--gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES')
    parser.add_argument('--verbose', type=int, default=logging.INFO,
                        help='Logging Level, 0, 10, ..., 50')
    parser.add_argument('--log_file', type=str, default='',
                        help='Logging file path')
    parser.add_argument('--result_file', type=str, default='',
                        help='Result file path')
    parser.add_argument('--result_folder', type=str, default='',
                        help='Result folder path')
    parser.add_argument('--random_seed', type=int, default=2021,
                        help='Random seed of numpy and pytorch.')
    parser.add_argument('--load', type=int, default=0,
                        help='Whether load model and continue to train')
    parser.add_argument('--train', type=int, default=1,
                        help='To train the model or not.')
    parser.add_argument('--finetune', type=int, default=0,
                        help='To finetune the model or not.')
    parser.add_argument('--eval', type=int, default=0,
                        help='To evaluate the model or not.')
    parser.add_argument('--time', type=str, default='',
                        help='Time to index model')
    parser.add_argument('--regenerate', type=int, default=0,
                        help='Whether to regenerate intermediate files.')
    parser.add_argument('--message', type=str, default='',
                        help='Additional message to add on the log/model name.')
    parser.add_argument('--meta_weighting', type=int, default=0,
                        help='Whether to use meta-model to reweight samples')
    return parser


def main():
    logging.info('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
    exclude = ['check_epoch', 'log_file', 'model_path', 'path', 'pin_memory',
               'regenerate', 'sep', 'train', 'verbose']
    logging.info(utils.format_arg_str(args, exclude_lst=exclude))

    # Random seed
    utils.fix_seed(args.random_seed)

    # GPU
    #os.environ["CUDA_VISIBLE_DEVICES"] = 'cpu'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logging.info('cuda available: {}'.format(torch.cuda.is_available()))
    logging.info('# cuda devices: {}'.format(torch.cuda.device_count()))

    # Read data
    # corpus_path = os.path.join(args.path, args.dataset, model_name.reader + '.pkl')
    corpus_path = os.path.join(args.path, args.dataset, args.suffix, args.fname, model_name.reader + '.pkl')
    if not args.regenerate and os.path.exists(corpus_path):
        logging.info('Load corpus from {}'.format(corpus_path))
        corpus = pickle.load(open(corpus_path, 'rb'))
        logging.info('Corpus loaded')
    else:
        corpus = reader_name(args)
        logging.info('Save corpus to {}'.format(corpus_path))
        pickle.dump(corpus, open(corpus_path, 'wb'))

    args.keys = ['train', 'test']
    corpus.n_train = int(corpus.n_batches*args.train_ratio)*args.batch_size
    corpus.n_test = corpus.dataset_size - corpus.n_train
    logging.info('Total instances: {}'.format(corpus.dataset_size))
    logging.info('Train instances: {}'.format(corpus.n_train))
    logging.info('Test instances: {}'.format(corpus.n_test))


    # Define model
    model = model_name(args, corpus)
    logging.info(model)
    model.apply(model.init_weights)
    model.actions_before_train()
    model.to(model._device)

    # Define meta-model
    if args.meta_weighting > 0:
        meta_model = meta_name(args, corpus, model)
        logging.info(meta_model)
        meta_model.to(model._device)
        meta_model.apply(meta_model.init_weights)

    else: meta_model = None

    # Run model
    data_dict = dict()
    for phase in ['train', 'test']:
        data_dict[phase] = model_name.Dataset(model, args, corpus, phase)
        data_dict[phase].g = copy.deepcopy(corpus.g.to(model._device))

    del corpus.g
    runner = runner_name(args)
    #logging.info('Test Before Training: ' + runner.print_res(model, data_dict, args, meta_model))
    if args.train > 0:
        runner.train(model, data_dict, args, meta_model)
    #logging.info(os.linesep + 'Test After Training: ' + runner.print_res(model, data_dict, args, meta_model))

    model.actions_after_train()
    logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)


if __name__ == '__main__':
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--model_name', type=str, default='BPR', help='Choose a model to run.')
    init_parser.add_argument('--meta_name', type=str, default='default', help='Choose a model to run.')
    init_args, init_extras = init_parser.parse_known_args()
    print(init_args.model_name)
    model_name = eval('{0}.{0}'.format(init_args.model_name))
    if not init_args.meta_name == 'default':
        meta_name = eval('{0}.{0}'.format(init_args.meta_name))
    reader_name = eval('{0}.{0}'.format(model_name.reader))
    runner_name = eval('{0}.{0}'.format(model_name.runner))


    # Args
    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    parser = reader_name.parse_data_args(parser)
    parser = runner_name.parse_runner_args(parser)
    parser = model_name.parse_model_args(parser)
    if not init_args.meta_name == 'default':
        parser = meta_name.parse_meta_args(parser)
    args, extras = parser.parse_known_args()
    if args.meta_weighting == 0:
        init_args.meta_name = 'default'

    # Logging configuration
    #log_args = [init_args.model_name, args.dataset, str(args.random_seed)]
    log_args = [init_args.model_name, args.dataset, str(args.random_seed), args.suffix, args.fname] # + str(args.test_length)]
    for arg in ['lr', 'l2'] + model_name.extra_log_args:
        log_args.append(arg + '=' + str(eval('args.' + arg)))
    log_file_name = '__'.join(log_args).replace(' ', '__')
    log_file_name += args.message
    if args.model_path == '':
        args.model_path = '../model/{}/{}_{}'.format(init_args.model_name, log_file_name, init_args.meta_name)
        args.meta_model_path = '../model/{}/{}_{}_meta'.format(init_args.model_name, log_file_name, init_args.meta_name)
    if args.train == 0 and args.meta_weighting != 0:
        args.extra_log = '__'.join(['meta-learning'+str(args.finetune), 'epoch'+str(args.epoch)])
        log_file_name += args.extra_log

    if args.load > 0 and args.time != '':
        time = args.time
    else:
        time = utils.get_time()
    log_file_name = '__'.join([log_file_name, init_args.meta_name, time]).replace(' ', '__')
    if args.log_file == '':
        args.log_file = '../log/{}/{}.txt'.format(init_args.model_name, log_file_name)
    if args.result_file == '':
        args.result_file = '../result/{}/{}'.format(init_args.model_name, log_file_name)
    if args.result_folder == '':
        args.result_folder = '../result/{}/{}/'.format(init_args.model_name, log_file_name)


    args.meta_name = init_args.meta_name
    args.model_name = init_args.model_name

    utils.check_dir(args.log_file)
    utils.check_dir(args.result_file)
    utils.check_dir(args.result_folder)
    files = os.listdir(args.result_folder)
    args.last_epoch = 0 if len(files)==0 else max([eval(epoch[:-4]) for epoch in files])
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(init_args)

    main()
