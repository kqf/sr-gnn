import torch
import pytest

from srgnn.modules import SessionGraph
from srgnn.experimental import SRGNN
from srgnn.dataset import batch_tensors


def _init(module):
    """Ensures deterministic weights
    """
    for weight in module.parameters():
        weight.data.fill_(1. / weight.shape[-1])
    return module


@pytest.fixture
def batch(batch_size=128, vocab_size=200, seq_len=12):
    seq = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size,))
    mask = ~torch.eq(seq, 0)
    x, y = batch_tensors(seq, mask, targets, device=torch.device("cpu"))
    return x


def test_modules(batch):
    original = _init(SessionGraph(100, 30000))
    reference = original(**batch)

    experimental = _init(SRGNN(100, 30000))
    output = experimental(**batch)

    assert torch.allclose(output, reference)
