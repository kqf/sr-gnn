import torch
import skorch
import numpy as np

from sklearn.pipeline import make_pipeline
from functools import partial

from srgnn.modules import SessionGraph
from srgnn.dataset import SequenceIterator, build_preprocessor


class DynamicVariablesSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        vocab = X.fields["text"].vocab
        net.set_params(module__vocab_size=len(vocab) + 1)
        # net.set_params(module__pad_idx=vocab["<pad>"])
        net.set_params(criterion__ignore_index=vocab["<pad>"])

        n_pars = self.count_parameters(net.module_)
        print(f'The model has {n_pars:,} trainable parameters')
        print(f'There number of unique items is {len(vocab)}')

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SeqNet(skorch.NeuralNet):
    def predict(self, X):
        # Now predict_proba returns top k indexes
        indexes = self.predict_proba(X)
        return np.take(X.fields["text"].vocab.itos, indexes)


def build_model(max_epochs=5, k=20):
    preprocessor = build_preprocessor(min_freq=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # TODO: Add LR-scheduler
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    model = skorch.NeuralNet(
        module=SessionGraph,
        module__hidden_size=100,
        module__vocab_size=30000,
        module__nonhybrid=True,
        module__step=1,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.001,
        optimizer__weight_decay=1e-5,
        criterion=torch.nn.CrossEntropyLoss,
        max_epochs=max_epochs,
        batch_size=100,
        iterator_train=SequenceIterator,
        # iterator_train__shuffle=True,
        # iterator_train__sort=False,
        iterator_valid=SequenceIterator,
        # iterator_valid__shuffle=False,
        # iterator_valid__sort=False,
        train_split=None,
        device=device,
        predict_nonlinearity=partial(inference, k=k, device=device),
        callbacks=[
            DynamicVariablesSetter(),
            skorch.callbacks.ProgressBar()
        ]
    )

    full = make_pipeline(
        preprocessor,
        model,
    )
    return full


def inference(logits, k, device):
    probas = torch.softmax(logits.to(device), dim=-1)
    # Return only indices
    return torch.topk(probas, k=k, dim=-1)[-1].clone().detach()
