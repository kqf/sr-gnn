import pytest
import torch

from srgnn.batch import batch_adjacency


@pytest.fixture
def sequence(batch_size, seq_size):
    return torch.randint(0, 5, (batch_size, seq_size))


@pytest.mark.parametrize("batch_size", [8, 32, 128, 137])
@pytest.mark.parametrize("seq_size", [8, 32, 128, 137])
def test_batch(sequence, batch_size, seq_size):
    alias, ain, aou = batch_adjacency(sequence)

    # Indices within the adjacency matrices
    assert alias.shape == (batch_size, seq_size)

    # Adjacency matrix of input edges
    assert ain.shape == (batch_size, seq_size, seq_size)

    # Adjacency matrix of output edges
    assert aou.shape == (batch_size, seq_size, seq_size)
