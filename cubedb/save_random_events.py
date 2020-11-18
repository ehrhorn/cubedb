import pickle
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def random_events_saver(dataset_name, limit=1e4):
    dataset_path = Path().home() / "work" / "datasets" / dataset_name
    dataset_db_file = dataset_path / "data" / f"{dataset_name}.db"
    dataset_uri = f"file:{str(dataset_db_file)}?mode=ro"
    output_file = dataset_path / "meta" / "powershovel_events.db"
    transformers = dataset_path / "meta" / "transformers.pkl"
    with open(transformers, "rb") as f:
        transformers = pickle.load(f)

    query = f"select * from truth order by random() limit {limit}"
    with sqlite3.connect(dataset_uri, uri=True) as con:
        truth = pd.read_sql(query, con)

    events_nos = truth["event_no"].values.flatten().tolist()
    stringified_event_nos = ", ".join([str(event_no) for event_no in events_nos])

    query = f"select * from features where event_no in ({stringified_event_nos})"
    with sqlite3.connect(dataset_uri, uri=True) as con:
        features = pd.read_sql(query, con)

    for column in truth.columns:
        if column in transformers["truth"]:
            transformer = transformers["truth"][column]
            data = truth[column].values.flatten().reshape(-1, 1)
            truth[column] = transformer.inverse_transform(data)
    for column in features.columns:
        if column in transformers["features"]:
            transformer = transformers["features"][column]
            data = features[column].values.flatten().reshape(-1, 1)
            features[column] = transformer.inverse_transform(data)

    with sqlite3.connect(str(output_file)) as con:
        truth.to_sql("truth", con, if_exists="replace", index=False)
        features.to_sql("features", con, if_exists="replace", index=False)

    indexing_queries = {}
    indexing_queries[
        "features_row"
    ] = """
        CREATE UNIQUE INDEX index_features ON features(row);
    """
    indexing_queries[
        "features_event_no"
    ] = """
        CREATE INDEX index_features_event_no ON features(event_no);
    """
    indexing_queries[
        "truth"
    ] = """
        CREATE UNIQUE INDEX index_truth ON truth(event_no);
    """

    for query in indexing_queries.values():
        with sqlite3.connect(str(output_file)) as con:
            cur = con.cursor()
            try:
                cur.execute(query)
            except Exception as e:
                print(str(e))


dataset_name = "dev_upgrade_train_step4_005"
random_events_saver(dataset_name, 1e4)
