import torch
import pytest
import random
import numpy as np

from srgnn.modules import SessionGraph
from srgnn.experimental import SRGNN, init_weights
from srgnn.dataset import batch_tensors

SEED = 137
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


@pytest.fixture
def batch(batch_size=128, vocab_size=100, seq_len=12):
    seq = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size,))
    mask = ~torch.eq(seq, 0)
    x, y = batch_tensors(seq, mask, targets, device=torch.device("cpu"))
    return x


def test_modules(batch):
    original = SessionGraph(100, 30000)
    init_weights(original)
    reference = original(**batch)

    experimental = SRGNN(100, 30000)
    init_weights(experimental)
    output = experimental(**batch)

    print(output - reference)
    # assert torch.allclose(output, reference)
