import torch
import pytest
from srgnn.modules import SessionGraph
from srgnn.dataset import batch_tensors


@pytest.fixture
def batch(batch_size=128, vocab_size=100, seq_len=12):
    seq = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size,))
    mask = ~torch.eq(seq, 0)
    x, y = batch_tensors(seq, mask, targets, device=torch.device("cpu"))
    return x


def test_modules(batch):
    model = SessionGraph(100, 30000)
    outputs = model(**batch)

    print(outputs)
