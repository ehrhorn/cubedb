import argparse
from datetime import datetime
from inspect import currentframe, getframeinfo
from pathlib import Path

from modules.sqlite_creator import create_sqlite_db


def get_dataset_paths(dataset_name: str):
    paths = {}
    home = Path().home()
    data_dir = Path().home().joinpath("data")
    fast_db = data_dir.joinpath(dataset_name + ".db")
    dataset_path = home.joinpath("work").joinpath("datasets").joinpath(dataset_name)
    data_path = dataset_path.joinpath("data")
    meta_path = dataset_path.joinpath("meta")
    data_path.mkdir(parents=True, exist_ok=True)
    meta_path.mkdir(parents=True, exist_ok=True)
    slow_db = data_path.joinpath(dataset_name + ".db")
    transformer_file = meta_path.joinpath("transformers.pkl")
    sets_file = meta_path.joinpath("sets.pkl")
    distributions_file = meta_path.joinpath("distributions.pkl")
    paths["fast_db"] = fast_db
    paths["db"] = slow_db
    paths["meta"] = meta_path
    paths["transformers"] = transformer_file
    paths["sets"] = sets_file
    paths["distributions"] = distributions_file
    errors_path = dataset_path.joinpath("errors")
    predictions_path = dataset_path.joinpath("predictions")
    errors_path.mkdir(parents=True, exist_ok=True)
    predictions_path.mkdir(parents=True, exist_ok=True)
    errors_db = errors_path.joinpath("errors.db")
    predictions_db = predictions_path.joinpath("predictions.db")
    paths["errors"] = errors_db
    paths["predictions"] = predictions_db
    return paths


parser = argparse.ArgumentParser()
parser.add_argument("-n")
parser.add_argument("-q")
args = parser.parse_args()

dataset_name = args.n
query_file = Path(__file__).parent.parent.resolve() / args.q

with open(query_file, "r") as f:
    query = f.read()

print(
    "{}: Beginning creating SQL db for dataset {}".format(datetime.now(), dataset_name)
)

create_sqlite_db(dataset_name, query)

print("{}: Ended creating SQL db for dataset {}".format(datetime.now(), dataset_name))
