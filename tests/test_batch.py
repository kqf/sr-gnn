import pytest
import numpy as np

from srgnn.batch import batch_adjacency


@pytest.fixture
def sequence(batch_size, seq_size):
    return np.random.randint(0, 5, (batch_size, seq_size))


@pytest.mark.parametrize("batch_size", [8, 32, 128, 137])
@pytest.mark.parametrize("seq_size", [8, 32, 128, 137])
def test_batch(sequence, batch_size, seq_size):
    a = batch_adjacency(sequence)
    assert a.shape == (batch_size, seq_size, seq_size * 2)
