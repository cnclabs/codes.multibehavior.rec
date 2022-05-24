#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import datetime
import math
import numpy as np
import torch
import copy
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from collections import defaultdict

class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class APANet(Module):
    def __init__(self, opt, n_item, n_act):
        super(APANet, self).__init__()
        self.hidden_size = opt.hidden_size
        self.n_item = n_item
        self.n_act = n_act
        self.batch_size = opt.batch_size
        self.item_embedding = nn.Embedding(self.n_item, self.hidden_size)
        self.action_embedding = nn.Embedding(self.n_act, self.hidden_size)
        self.pos_embedding1 = nn.Embedding(500, self.hidden_size) # (any size > max_session_len, hidden_size)
        
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=1, batch_first=True)
        
        self.dropout = torch.nn.Dropout(p=0.25)
        # Parameters
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.hidden_size, self.hidden_size))
        self.w_2 = nn.Parameter(torch.Tensor(self.hidden_size, 1))

        self.ilinear1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.ilinear2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.alinear1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.alinear2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.alinear3 = nn.Linear(self.hidden_size, 1, bias=True)
        self.alinear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        
        self.hlinear = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.leakyrelu = nn.LeakyReLU(0.2)

        self.loss_function = nn.CrossEntropyLoss()
        self.loss_function_act = nn.CrossEntropyLoss(weight=opt.weights)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def gru_seq_forward(self, action_seq_list):
        device_id = torch.cuda.current_device()
        device = f"cuda:{device_id}"
        seq_inputs = list()
        # get the length of each sentence
        seq_lengths = [len(seq) for seq in action_seq_list]
        # create an empty matrix with padding tokens
        padded_seq = np.zeros((len(action_seq_list), max(seq_lengths)))
        # copy over the actual sequences
        for i, seq_len in enumerate(seq_lengths):
            sequence = action_seq_list[i]
            padded_seq[i, 0:seq_len] = sequence[:seq_len]
        seq_inputs.append(padded_seq)

        act_seq_inputs = torch.as_tensor(padded_seq).int().to(device)
        act_seq_inputs = self.action_embedding(act_seq_inputs)
        act_seq_inputs = torch.nn.utils.rnn.pack_padded_sequence(act_seq_inputs, seq_lengths, batch_first=True, enforce_sorted=False)
        seq_outputs, seq_hidden = self.gru(act_seq_inputs, None)
        return seq_outputs, seq_hidden

    def next_item_predict(self, seq_hidden, action_hidden, mask, itemindexTensor, actionindexTensor):
        action_hidden = self.dropout(action_hidden)
        hidden = seq_hidden + action_hidden
        # add position embedding
        pos_emb = self.pos_embedding1.weight[:hidden.shape[1]]
        pos_emb = pos_emb.unsqueeze(0).repeat(hidden.shape[0], 1, 1)
        mask = mask.float().unsqueeze(-1)
        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, hidden.shape[1], 1)
        hk = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        hk = torch.tanh(hk)
        hk = torch.sigmoid(self.ilinear1(hk) + self.ilinear2(hs))
        alpha = torch.matmul(hk, self.w_2)
        alpha = alpha * mask
        a = torch.sum(alpha * hidden, 1)
        b = self.item_embedding.weight[1:]  # n_item x latent_size
        item_scores = torch.matmul(a, b.transpose(1, 0))
        A = self.action_embedding.weight[actionindexTensor]  # n_acts *latent_size

        return item_scores, A
    
    # conditional next action prediction
    def next_action_predict(self, action_hidden, mask, A, next_item, item_actseq, test=False, idx=None):        
        # next item hidden
        pred_i_ht = self.item_embedding.weight[next_item] # next item's embedding # train -> target (ground truth)

        # retrieve correspond act seq hidden
        act_seq = list()
        for i, it in enumerate(next_item):
            if test: # check the same index for top n items
                i = idx 
            if it in item_actseq[i]:
                pred_i_actseq = item_actseq[i][it]
                act_seq.append(pred_i_actseq)
            else:
                act_seq.append([0])

        last_seq_outputs, last_seq_hidden = self.gru_seq_forward(act_seq) # (1, batch_size, hidden_size)
        last_seq_hidden =  last_seq_hidden.squeeze(0) #torch.swapaxes(last_seq_hidden, 0, 1)
        next_ht = pred_i_ht + last_seq_hidden
        
        ni_ht = self.alinear1(next_ht).view(next_ht.shape[0], 1, next_ht.shape[1])  # batch_size x 1 x latent_size
        as_ht = self.alinear2(action_hidden)  # batch_size x seq_length x latent_size
        beta = self.alinear3(torch.sigmoid(ni_ht + as_ht))
        sh = torch.sum(beta * action_hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        
        s_act = self.alinear_transform(torch.cat([sh, next_ht], 1))
        action_scores = torch.matmul(s_act, A.transpose(1, 0))
        return action_scores

    def next_action_predict_test(self, action_hidden, mask, A, item_scores, item_actseq):
        top_act_scores = []
        pred_top20_i = item_scores.topk(20, 1)[1]

        for i, top_items in enumerate(pred_top20_i):
            # get the current session and duplicate 20 times for each
            curr_action_hidden = action_hidden[i].repeat(20, 1, 1)
            curr_mask = mask[i].repeat(20, 1)
            action_scores = self.next_action_predict(curr_action_hidden, curr_mask, A, top_items, item_actseq, test=True, idx=i)
            top_act_scores.append(action_scores)
            
        return top_act_scores

    def test(self, n, L):
        L.append(n)

    def forward(self, inputs, A, action_inputs, item_inputs):
        item_hidden = self.item_embedding(inputs)
        item_hidden = self.gnn(A, item_hidden)
        
        # get action sequence under a item
        seq_hidden_list = list()
        # batch-level
        item_actseq = []
        for i_seq, a_seq in zip(item_inputs.tolist(), action_inputs.tolist()):
            tracking = defaultdict(list)
            action_seq_list = list()
            # sequence-level
            for i, a in zip(i_seq, a_seq):
                tracking[i].append(a)
                action_seq_list.append(tracking[i])
            # save for next action prediction
            if 0 in tracking:
                tracking.pop(0)
            item_actseq.append(tracking)

            seq_outputs, seq_hidden = self.gru_seq_forward(action_seq_list)
            seq_hidden_list.append(seq_hidden)

        seq_hiddens = torch.cat(seq_hidden_list) 

        action_inputs = self.action_embedding(action_inputs)
        action_output, action_hidden = self.gru(action_inputs,None)
        ## two kinds of action emb
        seq_hiddens = torch.cat([seq_hiddens, action_output], -1)
        seq_hiddens = self.hlinear(seq_hiddens)
        seq_hiddens = self.leakyrelu(seq_hiddens)
        return item_hidden, seq_hiddens, item_actseq


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

def matrixtopk(score, k=20):
    v, i = torch.topk(score.flatten(), k)
    return np.array( np.unravel_index(i.cpu(), score.shape)).T.tolist()

# multi-task learning
def loss_function(model, item_scores, targets, action_scores, action_targets, l=0.5):
    item_loss = model.loss_function(item_scores, targets)
    action_loss = model.loss_function_act(action_scores, action_targets)
    loss = item_loss + l*action_loss
    return loss

def forward_model(model, i, data, test=False):
    _, _, itemindexTensor, actionindexTensor = data.get_data_info()
    alias_inputs, A, items, action_inputs, mask, targets, action_targets, item_inputs = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    action_inputs = trans_to_cuda(torch.Tensor(action_inputs).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    item_hidden, action_hidden, item_actseq = model(items, A, action_inputs, item_inputs)
    get = lambda i: item_hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    item_scores, A = model.next_item_predict(seq_hidden, action_hidden, mask, itemindexTensor, actionindexTensor) 
    if test:
        action_scores = model.next_action_predict_test(action_hidden, mask, A, item_scores, item_actseq)
    else:
        action_scores = model.next_action_predict(action_hidden, mask, A, targets, item_actseq)
    return item_scores, targets, action_scores, action_targets


def train_(model, train_data, l):
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    model.train()
    hit, mrr = [], []
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        itemid2index, actionid2index, _, _ = train_data.get_data_info()
        item_scores, targets, action_scores, action_targets = forward_model(model, i, train_data)
        targets = [itemid2index[tar] for tar in targets]
        targets = trans_to_cuda(torch.Tensor(targets).long())
        action_targets = [ actionid2index[a_t]  for a_t in action_targets]
        action_targets = trans_to_cuda(torch.Tensor(action_targets).long())
        loss = loss_function(model, item_scores, targets, action_scores, action_targets, l)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % 100 == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()),datetime.datetime.now())
        
        ## compute performance
        sub_scores = item_scores.topk(20)[1]  # tensor has the top_k functions
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target in zip(sub_scores, trans_to_cpu(targets).detach().numpy()):
            hit.append(np.isin(target, score))
            if len(np.where(score == target)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target)[0][0] + 1))

    model.scheduler.step()
    print('\tLoss:\t%.3f' % total_loss)
    print('Hit', np.mean(hit) * 100, 'MRR', np.mean(mrr) * 100)
    return total_loss

def predict_(model, test_data):
    print('start predicting: ', datetime.datetime.now())
    n_act = test_data.n_act
    itemid2index, actionid2index, _, _ = test_data.get_data_info()
    with torch.no_grad():
        hit, mrr, IA_hit, IA_mrr = [], [], [], []
        slices = test_data.generate_batch(model.batch_size)
        for i in slices:
            item_scores, targets, action_scores, action_targets = forward_model(model, i, test_data, test=True)
            sub_scores = item_scores.topk(20)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            targets = [itemid2index[tar] for tar in targets]
            action_targets = [actionid2index[a_t]  for a_t in action_targets]
            item_action_targets = [(i*n_act) + a for i, a in zip(targets, action_targets)]

            # next item prediction
            for score, target in zip(sub_scores, targets):
                hit.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr.append(0)
                else:
                    mrr.append(1 / (np.where(score == target)[0][0] + 1))
            
            # next item & action prediction
            for top_items, act_score, IA_target in zip(sub_scores, action_scores, item_action_targets):
                top_act = torch.argmax(act_score, 1)
                pred = top_items * n_act + torch.argmax(act_score, 1).cpu().numpy()
                IA_hit.append(np.isin(IA_target, pred))
                if len(np.where(pred == IA_target)[0]) == 0:
                    IA_mrr.append(0)
                else:
                    IA_mrr.append(1 / (np.where(pred == IA_target)[0][0] + 1))

        hit = np.mean(hit) * 100
        mrr = np.mean(mrr) * 100
        IA_hit = np.mean(IA_hit) * 100
        IA_mrr = np.mean(IA_mrr) * 100
        return hit, mrr, IA_hit, IA_mrr
