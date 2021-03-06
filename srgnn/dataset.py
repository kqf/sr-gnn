import warnings

import torch
from sklearn.base import BaseEstimator, TransformerMixin
from torchtext.data import Example, Dataset, Field, BucketIterator

from srgnn.batch import batch


class PandasDataset(Dataset):
    def __init__(self, df, fields):
        # Fix for scorch sparsity check issues
        self.is_sparse = False
        proc = [df[col].apply(f.preprocess) for col, f in fields]
        examples = [Example.fromlist(f, fields) for f in zip(*proc)]
        super().__init__(examples, fields)


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, fields, min_freq=1):
        self.fields = fields
        self.min_freq = min_freq

    def fit(self, X, y=None):
        dataset = self.transform(X, y)
        for name, field in dataset.fields.items():
            if field.use_vocab:
                field.build_vocab(dataset, min_freq=self.min_freq)
        return self

    def transform(self, X, y=None):
        with warnings.catch_warnings(record=True):
            return PandasDataset(X, self.fields)


def build_preprocessor(min_freq=5):
    with warnings.catch_warnings(record=True):
        text_field = Field(
            tokenize=None,
            init_token=None,
            pad_token="<unk>",
            unk_token="<unk>",
            eos_token=None,
            batch_first=True,
            # pad_first=True,
        )
        fields = [
            ('text', text_field),
            ('gold', text_field),
        ]
        return TextPreprocessor(fields, min_freq=min_freq)


class SequenceIterator(BucketIterator):
    def __init__(self, *args, **kwargs):
        with warnings.catch_warnings(record=True):
            super().__init__(*args, **kwargs)

    def __iter__(self):
        pi = self.dataset.fields["text"].vocab.stoi["<pad>"]
        with warnings.catch_warnings(record=True):
            for batch_ in super().__iter__():
                mask = ~torch.eq(batch_.text, pi)
                seq, target = batch_.text, batch_.gold.view(-1)
                yield batch(seq, mask, target, self.device)


def train_split(X, prep, X_val):
    if X_val is None:
        train, validation = Dataset.split(X)
        # Fix for skorch sparsity checks
        train.is_sparse = False
        validation.is_sparse = False
        return train, validation

    return X, prep.transform(X_val)
