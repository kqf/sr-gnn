import click
from srgnn.data import read_data, ev_data

from srgnn.model import build_model


@click.command()
@click.option(
    "--path", type=click.Path(exists=True), default="data/processed/")
def main(path):
    train, test, valid = read_data(path)
    data = ev_data(train["text"])

    model = build_model()
    model.fit(data)


if __name__ == '__main__':
    main()
