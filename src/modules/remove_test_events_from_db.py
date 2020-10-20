from datetime import datetime
from pathlib import Path
import sqlite3

import pandas as pd

db = Path().home().joinpath("data").joinpath("dev_upgrade_000.db")
csv_file = Path().home().joinpath("train_set.csv")

train_event_nos = pd.read_csv(csv_file)
train_event_nos = train_event_nos[train_event_nos["run_type"] == 4][
    "event_no"
].values.tolist()

with sqlite3.connect(db) as con:
    query = "select event_no from truth"
    truth = pd.read_sql(query, con)

event_nos = truth["event_no"].values.tolist()
valid_event_nos = list(set(event_nos) - set(train_event_nos))
# valid_event_nos = [str(event_no) for event_no in valid_event_nos]

with sqlite3.connect(db) as con:
    cur = con.cursor()
    query = "delete from features where event_no in ({})".format(
        ", ".join([str(event_no) for event_no in valid_event_nos])
    )
    cur.execute(query)
    query = "delete from truth where event_no in ({})".format(
        ", ".join([str(event_no) for event_no in valid_event_nos])
    )
    cur.execute(query)
    con.commit()

with sqlite3.connect(db) as con:
    cur = con.cursor()
    query = "select count(event_no) from truth"
    cur.execute(query)
    count = cur.fetchall()

print(count)