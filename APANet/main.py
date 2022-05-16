#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import argparse
import pickle
import time
from utils import Data, split_validation, seed_torch, save_results
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: Rrocket/Kkbox/yoochoose/sample')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=50, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-6, help='l2 penalty')
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=5, help='the number of epoch to wait before early stop ')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--l', type=float, default=-1, help='parameter (lambda) in multi-task loss function')
parser.add_argument('--seed', type=int, default=2021, help='random seed')
opt = parser.parse_args()
print(opt)


def main():
    data_path = 'datasets/' + opt.dataset

    info = pickle.load(open(data_path + '/info.pkl', 'rb'))
    n_item, n_act = info[0]+1, info[1]+1
    print('n_item:{}, n_action:{}'.format(n_item-1, n_act-1))

    train_data = pickle.load(open(data_path + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open(data_path + '/test.txt', 'rb'))

    train_data = Data(train_data, info, shuffle=True)
    test_data = Data(test_data, info, shuffle=False)

    # action loss weight
    label, count = np.unique(train_data.data_action_targets, return_counts=True)
    opt.weights = torch.FloatTensor([1 - (i / sum(count)) for i in count]) # w=(1-class_n/total_n) 

    # multi-task loss param (lambda)
    lamda = {'Kkbox': 10, 'Rrocket': 1, 'yoochoose': 1, 'sample': 1}
    if opt.l == -1:
        opt.l = lamda[opt.dataset] 

    seed_torch(int(opt.seed))
    model = trans_to_cuda(MB_SR(opt, n_item, n_act))
    results = []
    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    best_result_ia = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        total_loss = train_(model, train_data, opt.l)
        hit, mrr, IA_hit, IA_mrr = predict_(model, test_data)

        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            best_result_ia = [IA_hit, IA_mrr]
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            best_result_ia = [IA_hit, IA_mrr]
            flag = 1
        if IA_hit >= best_result_ia[0]:
            best_result_ia = [IA_hit, IA_mrr]

        print('Recall@20:\t%.4f\tMMR@20:\t%.4f\tIA_hit@20:\t%.4f\tIA_mrr@20:\t%.4f\t' % (hit, mrr, IA_hit, IA_mrr))
        
        print('Best Result:')
        print('Recall@20:\t%.4f\tMMR@20:\t%.4f\tIA_hit@20:\t%.4f\tIA_mrr@20:\t%.4f\t' % (best_result[0], best_result[1], best_result_ia[0], best_result_ia[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    
    # record results
    save_results(np.array([best_result+best_result_ia]), opt.dataset)

    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))
    


if __name__ == '__main__':
    main()
