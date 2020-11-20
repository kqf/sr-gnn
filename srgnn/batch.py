import torch


def ix(a, b, seq_size):
    """ Construct indices in adjacency matrices within batch dimension
    from lists of vectors:

    ix([batch_size, m], [batch_size, m]) -> idx
    Where idx indexes a matrix of shape [batch_size, seq_size, seq_size]

    Attributes:
        a: A matrix that contains indices of shape [batch_size, slen]
        where slen < seq_len.
        b: A matrix that contains indices of shape [batch_size, slen]
        where slen < seq_len.
        seq_size: is the size of the desired adjacency matrix

    """
    batch_size, _ = a.shape
    idx_per_batch = a + seq_size * b
    batch_offset = seq_size * seq_size * torch.arange(batch_size)
    return idx_per_batch + batch_offset[:, None]


def batch_adjacency(sequence):
    # Replace ids to indexes inside the adjacency matrix
    eq = sequence[:, :, None] == sequence[:, None, :]
    aliases = eq.long().argmax(1)

    batch_size, seq_size = sequence.shape
    asize = (batch_size, seq_size, seq_size)

    # Calculate the incoming edges
    inp = torch.zeros(asize).view(-1)
    inp[ix(aliases[:, :-1], aliases[:, 1:], seq_size)] = 1.
    inp = inp.view(*asize)

    # Calculate the outgoing edges
    out = torch.zeros(asize).view(-1)
    out[ix(aliases[:, 1:], aliases[:, :-1], seq_size)] = 1.
    out = out.view(*asize)

    # Concatenate as in the original implementation
    return aliases, inp, out


def batch(seq, mask, target, device):
    alias_inputs, ain, aou = batch_adjacency(seq)

    batch = {}
    batch["alias_inputs"] = torch.tensor(alias_inputs).to(device)
    batch["items"] = torch.tensor(seq).to(device)
    batch["ain"] = torch.tensor(ain).float().to(device)
    batch["aou"] = torch.tensor(aou).float().to(device)
    batch["mask"] = torch.tensor(mask).to(device)
    target = torch.tensor(target).to(device)
    return batch, target
