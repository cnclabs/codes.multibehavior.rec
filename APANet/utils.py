#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import numpy as np
import math
import torch
import os
import random

# Seed
def seed_torch(seed=2021):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def save_results(results, t=''):
    if not os.path.exists('results'):
        os.makedirs('results')
    r_file = 'results/'+t+'.csv'
    if not os.path.exists(r_file):
        np.savetxt(r_file, np.array(results), delimiter=',', fmt='%s')
    else:
        with open(r_file, 'a') as f:
            np.savetxt(f, np.array(results), delimiter=',', newline='\n', fmt='%s')

def getMask(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks

def split_validation(train_set, valid_portion):
    n_samples = len(train_set[0])
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    temp_arr = np.asarray(train_set, dtype=object)
    train_, valid_ = tuple(temp_arr[:, :n_train].tolist()), tuple(temp_arr[:, n_train:].tolist())
    return train_, valid_

class Data():
    def __init__(self, data, info, shuffle=False):
        inputs, mask = getMask(data[0], [0])
        act, _ = getMask(data[2], [0])
        self.data_paddings = np.asarray(inputs)
        self.data_masks = np.asarray(mask)
        self.data_action_paddings = np.asarray(act)
        self.data_targets = np.asarray(data[1])
        self.data_action_targets = np.asarray(data[3])
        self.n_item, self.n_act = info[0]+1, info[1]+1
        self.length = len(inputs)
        self.shuffle = shuffle
    
    def get_data_info(self):
        item_ids, action_ids = np.array(list(range(1, self.n_item))), np.array(list(range(1, self.n_act)))
        itemid2index, actionid2index = {}, {}
        for index, item_id in enumerate(item_ids):
            itemid2index[item_id] = index
        for index, action_id in enumerate(action_ids):
            actionid2index[action_id] = index
        itemindexTensor = torch.Tensor(item_ids).long()
        actionindexTensor = torch.Tensor(action_ids).long()
        return itemid2index, actionid2index, itemindexTensor, actionindexTensor

    def generate_batch(self, batch_size):
        n_batch = math.ceil(self.length / batch_size)
        shuffle_args = np.arange(n_batch*batch_size)
        if self.shuffle:
            np.random.shuffle(shuffle_args)
        slices = np.split(shuffle_args,n_batch)
        slices = [i[i<self.length] for i in slices]
        return slices

    def get_slice(self, i):
        item_inputs, action_inputs, masks = self.data_paddings[i], self.data_action_paddings[i], self.data_masks[i]
        targets, action_targets = self.data_targets[i], self.data_action_targets[i]
        items, n_node, A, alias_input = [], [], [], []
        for u_input in item_inputs:
            n_node.append(len(np.unique(u_input))) #the length of unique items
        max_n_node = np.max(n_node) #the longest unique item length
        for u_input in item_inputs:
            node = np.unique(u_input) #the unique items of inputs
            items.append(node.tolist()+(max_n_node-len(node))*[0]) #items list
            u_A = np.zeros((max_n_node,max_n_node))
            for i in range(len(u_input)-1):
                if u_input[i+1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0] #np.where return a tuple,so need use [0][0] to show the value
                v = np.where(node == u_input[i+1])[0][0]
                u_A[u][v] +=1
            u_sum_in = np.sum(u_A,0) # in degree
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A,u_sum_in)
            u_sum_out = np.sum(u_A,1) #out degree
            u_sum_out[np.where(u_sum_out ==0)] = 1
            u_A_out = np.divide(u_A.T,u_sum_out)
            u_A = np.concatenate([u_A_in,u_A_out]).T
            A.append(u_A)
            alias_input.append([np.where(node == i)[0][0] for i in u_input] )
        
        return alias_input, A, items, action_inputs, masks, targets, action_targets, item_inputs