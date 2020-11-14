#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import datetime
import math
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F


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


class SessionGraph(Module):
    def __init__(self, hidden_size, n_node, nonhybrid=True, step=1):
        super(SessionGraph, self).__init__()
        self.hidden_size = hidden_size
        self.n_node = n_node
        self.nonhybrid = nonhybrid

        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def _forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden

    def forward(self, alias_inputs, A, items, mask):
        hidden = self._forward(items, A)
        get = lambda i: hidden[i][alias_inputs[i]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        return self.compute_scores(seq_hidden, mask)



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


def train_test(model, trainiterator, testiterator):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    for j, (batch, targets) in enumerate(trainiterator):
        model.optimizer.zero_grad()
        scores = model(**batch)
        loss = model.loss_function(scores, targets)
        loss.backward()
        model.optimizer.step()
        model.scheduler.step()
        total_loss += loss
        if j % int(len(batch["items"]) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (
                j, len(batch["items"][0]), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    # model.eval()
    # hit, mrr = [], []
    # for batch in testiterator:
    #     scores = forward(*batch)
    #     sub_scores = scores.topk(20)[1]
    #     sub_scores = trans_to_cpu(sub_scores).detach().numpy()
    #     for score, target, mask in zip(sub_scores, targets, test_data.mask):
    #         hit.append(np.isin(target - 1, score))
    #         if len(np.where(score == target - 1)[0]) == 0:
    #             mrr.append(0)
    #         else:
    #             mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    # hit = np.mean(hit) * 100
    # mrr = np.mean(mrr) * 100
    # return hit, mrr