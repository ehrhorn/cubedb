from datetime import datetime
from pathlib import Path
import sqlite3

import pandas as pd
from sklearn.model_selection import train_test_split


meta_db = Path().home().joinpath("data").joinpath("meta.db")

query = """
    select event_no, run_type from meta
"""

print("{}: Reading events and run types...".format(datetime.now()))

with sqlite3.connect(str(meta_db)) as con:
    df = pd.read_sql(query, con)

train_event_nos = []

print("{}: Splitting...".format(datetime.now()))

out_df = pd.DataFrame(columns=["event_no", "run_type"])

for run_type in df["run_type"].unique():
    if run_type == 1:
        continue
    else:
        event_nos = df[df["run_type"] == run_type]["event_no"].values
        train, _ = train_test_split(event_nos, test_size=0.2, random_state=29897070)
        temp_df = pd.DataFrame(data={"event_no": train, "run_type": run_type})
        out_df = out_df.append(temp_df, ignore_index=True)

print("{}: Saving csv...".format(datetime.now()))
out_df.to_csv(Path().home().joinpath("train_set.csv"))

print("{}: Done!".format(datetime.now()))