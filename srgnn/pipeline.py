import torch
import skorch

from sklearn.pipeline import make_pipeline

from model.tmodel import SessionGraph
from model.dataset import SequenceIterator, build_preprocessor


def build_model(opt, n_node):
    preprocessor = build_preprocessor(min_freq=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = skorch.NeuralNet(
        module=SessionGraph,
        module__opt=opt,
        module__n_node=n_node,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.002,
        criterion=torch.nn.CrossEntropyLoss,
        max_epochs=5,
        batch_size=512,
        iterator_train=SequenceIterator,
        iterator_train__shuffle=True,
        iterator_train__sort=False,
        iterator_valid=SequenceIterator,
        iterator_valid__shuffle=False,
        iterator_valid__sort=False,
        train_split=None,
        device=device,
    )

    full = make_pipeline(
        preprocessor,
        model,
    )
    return full
