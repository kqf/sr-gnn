import torch
import skorch

from sklearn.pipeline import make_pipeline

from srgnn.modules import SessionGraph
from srgnn.dataset import SequenceIterator, build_preprocessor


def build_model(opt, n_node):
    preprocessor = build_preprocessor(min_freq=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # TODO: Add LR-scheduler
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    model = skorch.NeuralNet(
        module=SessionGraph,
        module__hidden_size=100,
        module__n_node=n_node,
        module__nonhybrid=True,
        module__step=1,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.001,
        optimizer__weight_decay=1e-5,
        criterion=torch.nn.CrossEntropyLoss,
        max_epochs=5,
        batch_size=100,
        iterator_train=SequenceIterator,
        # iterator_train__shuffle=True,
        # iterator_train__sort=False,
        iterator_valid=SequenceIterator,
        # iterator_valid__shuffle=False,
        # iterator_valid__sort=False,
        train_split=None,
        device=device,
        callbacks=[
            skorch.callbacks.ProgressBar()
        ]
    )

    full = make_pipeline(
        preprocessor,
        model,
    )
    return full
