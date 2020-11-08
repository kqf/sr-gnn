import click
from model.data import read_data, ev_data
from model.utils import Data


@click.command()
@click.option(
    "--path", type=click.Path(exists=True), default="data/processed/")
def main(path):
    train, test, valid = read_data(path)
    data = ev_data(train["text"])

    raw_data = (data["text"], data["gold"])
    dataset = Data(raw_data)


if __name__ == '__main__':
    main()
