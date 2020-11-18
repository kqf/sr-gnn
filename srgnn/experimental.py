import torch
from torch.nn import Parameter
import torch.nn.functional as F


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
    offset = torch.arange(eshape[0], device=device) * (eshape[1])

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

        self._ein = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        self._eou = torch.nn.Linear(hidden_size, hidden_size, bias=True)

    def cell(self, A, hidden):
        input_in = torch.matmul(
            A[:, :, :A.shape[1]], self._ein(hidden)) + self.b_iah

        input_out = torch.matmul(
            A[:, :, A.shape[1]:], self._eou(hidden)) + self.b_oah

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
            hidden = self.cell(A, hidden)
        return hidden


class SRGNN(torch.nn.Module):
    def __init__(self, hidden_size, vocab_size, nonhybrid=True, step=1):
        super(SRGNN, self).__init__()
        self.nonhybrid = nonhybrid
        self._emb = torch.nn.Embedding(vocab_size, hidden_size)
        self._gnn = GNN(hidden_size, step=step)
        self._fc1 = torch.nn.Linear(hidden_size, hidden_size)
        self._fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self._fc3 = torch.nn.Linear(hidden_size, 1, bias=False)
        self._fcp = torch.nn.Linear(hidden_size * 2, hidden_size)

    def _scores(self, hidden, mask):
        # [batch, hidden]
        # TODO: Switch to the last item
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]

        # [batch, 1, hidden]
        q1 = self._fc1(ht).unsqueeze(1)

        # [batch, seq, hidden]
        q2 = self._fc2(hidden)

        alpha = self._fc3(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.unsqueeze(-1), 1)
        if not self.nonhybrid:
            a = self._fcp(torch.cat([a, ht], 1))

        # [batch, hidden] @ [vocab_size x hidden].T
        return a @ self._emb.weight[1:].T

    def _embed(self, A, items):
        emb = self._emb(items)
        hidden = self._gnn(A, emb)
        return hidden

    def forward(self, alias_inputs, A, items, mask):
        # Use GNNs to exploit graph structure of the session
        # items are needed only to extract features

        # hidden [batch, seq, hidden]
        hidden = self._embed(A, items)

        # Use alias_inputs indexes to use the propagated embeddings
        # seq [batch, seq, hidden]
        embeddings = batch_emb(hidden, alias_inputs)

        # Calculate the logprobs for the next items
        return self._scores(embeddings, mask)
