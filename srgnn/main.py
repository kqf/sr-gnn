import click
from srgnn.data import read_data, ev_data

from srgnn.model import build_model, evaluate


@click.command()
@click.option(
    "--path", type=click.Path(exists=True), default="data/processed/")
def main(path):
    train, test, valid = read_data(path)
    X = ev_data(train["text"])
    X_val = ev_data(train["text"])
    X_te = ev_data(train["test"])

    model = build_model(X_val=X_val)
    model.fit(X)

    evaluate(model, X, "train")
    evaluate(model, X_val, "valid")
    evaluate(model, X_te, "test")


if __name__ == '__main__':
    main()
