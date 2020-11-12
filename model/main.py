import click
from collections import namedtuple
from model.data import read_data, ev_data
from model.utils import Data
from model.tmodel import trans_to_cuda, SessionGraph, train_test


@click.command()
@click.option(
    "--path", type=click.Path(exists=True), default="data/processed/")
def main(path):
    train, test, valid = read_data(path)
    data = ev_data(train["text"])

    raw_data = (data["text"], data["gold"])
    dataset = Data(raw_data)

    n_node = 43098

    optf = namedtuple(
        "opt", [
            "hiddenSize",
            "batchSize",
            "nonhybrid",
            "step",
            "lr",
            "l2",
            "lr_dc_step",
            "lr_dc"
        ])

    opt = optf(
        hiddenSize=100,
        batchSize=100,
        nonhybrid=True,
        step=1,
        l2=1e-5,
        lr=0.001,
        lr_dc_step=3,
        lr_dc=0.1
    )

    model = trans_to_cuda(SessionGraph(opt, n_node))
    train_test(model, dataset, dataset)


if __name__ == '__main__':
    main()
