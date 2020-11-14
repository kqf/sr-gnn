import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


def init_weights(module):
    for weight in module.parameters():
        weight.data.fill_(1. / weight.shape[-1])


def batch_emb(emb, x):
    """ Embeds x with matrix emb along batch dimension

    In other words converts the lookup in [batch, seq, hidden]
    to [batch * seq, hidden]

    Assuming that x is of the shape [batch, seq]
    """
    eshape = emb.shape
    # embs [batch * seq, hidden]
    embs = emb.view(-1, eshape[-1])

    device = embs.device
    offset = torch.arange(eshape[0], device=device) * (x.shape[1])

    # Convert sequence of indexes to indexes in a new matrix
    idx = x + offset.unsqueeze(-1)
    return embs[idx]


class GNN(torch.nn.Module):
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

        self.linear_edge_in = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]],
                                self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(
            A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
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


class SRGNN(torch.nn.Module):
    def __init__(self, hidden_size, n_node, nonhybrid=True, step=1):
        super(SRGNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_node = n_node
        self.nonhybrid = nonhybrid

        self._emb = nn.Embedding(self.n_node, self.hidden_size)
        self._gnn = GNN(self.hidden_size, step=step)
        self._fc1 = torch.nn.Linear(hidden_size, hidden_size)
        self._fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self._fc3 = torch.nn.Linear(hidden_size, 1)
        self._fcp = torch.nn.Linear(hidden_size * 2, hidden_size)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(
            mask, 1) - 1]  # batch_size x latent_size
        # batch_size x 1 x latent_size
        q1 = self._fc1(ht).view(ht.shape[0], 1, ht.shape[1])
        q2 = self._fc2(hidden)  # batch_size x seq_length x latent_size
        alpha = self._fc3(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden *
                      mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self._fcp(torch.cat([a, ht], 1))
        b = self._emb.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def _forward(self, inputs, A):
        emb = self._emb(inputs)
        hidden = self._gnn(A, emb)
        return hidden

    def forward(self, alias_inputs, A, items, mask):
        # [batch_size, seq_len, hidden_size]
        hidden = self._forward(items, A)
        seq = batch_emb(hidden, alias_inputs)
        return self.compute_scores(seq, mask)
