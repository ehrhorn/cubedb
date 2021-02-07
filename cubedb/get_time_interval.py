import pickle
import sqlite3
from pathlib import Path

import pandas as pd

dataset_name = "dev_genie_numu_cc_train_retro_005"
datasets_path = Path().home() / "work" / "erda_datasets"
sets_path = datasets_path / dataset_name / "meta" / "sets.pkl"
db_path = Path().home() / "work" / "datasets" / "meta.db"
db_uri = f"file:{db_path}?mode=ro"
with open(sets_path, "rb") as f:
    sets = pickle.load(f)

set_name = "test"
event_nos = sets[set_name]
stringified_event_nos = ", ".join([str(event_no) for event_no in event_nos])

query = f"select unix_start_time from meta where event_no in ({stringified_event_nos})"
with sqlite3.connect(db_uri, uri=True) as con:
    times = pd.read_sql(query, con)

print(times.max() - times.min())
