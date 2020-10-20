from datetime import datetime
from datetime import timedelta
from functools import update_wrapper
import itertools
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


meta_db = Path().home().joinpath("data").joinpath("meta_2.db")

query = """
    select event_no, run_type from meta
"""

print("{}: Reading events and run types...".format(datetime.now()))

with sqlite3.connect(str(meta_db)) as con:
    df = pd.read_sql(query, con)

train_event_nos = []

print("{}: Splitting...".format(datetime.now()))

for run_type in df["run_type"].unique():
    if run_type == 1:
        continue
    else:
        event_nos = df[df["run_type"] == run_type]["event_no"].values
        train, _ = train_test_split(event_nos, test_size=0.2, random_state=29897070)
        train_event_nos.extend(train)

print("{}: Updating column...".format(datetime.now()))

n = 100000
chunked_train_event_nos = list(divide_chunks(train_event_nos, n))
deltas = []

with sqlite3.connect(str(meta_db)) as con:
    cur = con.cursor()
    try:
        cur.execute("alter table meta add column train integer default 0")
    except Exception as e:
        print("{}: Exception encountered: {}".format(datetime.now(), str(e)))

for iter, chunk in enumerate(chunked_train_event_nos):
    start = datetime.now()
    with sqlite3.connect(str(meta_db)) as con:
        cur = con.cursor()
        query = "select event_no from meta where event_no in ({}) and train = 0".format(
            ", ".join([str(event_no) for event_no in chunk])
        )
        cur.execute(query)
        valid_event_nos = cur.fetchall()
        valid_event_nos = [event_no[0] for event_no in valid_event_nos]
        if valid_event_nos:
            query = "update meta set train = 1 where event_no in ({})".format(
                ", ".join([str(event_no) for event_no in valid_event_nos])
            )
            cur.execute(query)
            con.commit()
    end = datetime.now()
    delta = (end - start).total_seconds()
    deltas.append(delta)
    avg_time = sum(deltas) / len(deltas)
    chunks_left = len(chunked_train_event_nos) - (iter + 1)
    eta_seconds = avg_time * chunks_left
    eta_min, eta_sec = divmod(eta_seconds, 60)
    eta_hour, eta_min = divmod(eta_min, 60)
    eta_time = datetime.now() + timedelta(seconds=eta_seconds)
    print(
        "{}: {} chunks down, {} to go, ETA {:02d}:{:02d}:{:02d}; that's {}".format(
            datetime.now(),
            iter + 1,
            chunks_left,
            int(eta_hour),
            int(eta_min),
            int(eta_sec),
            eta_time,
        )
    )

print("{}: Done!".format(datetime.now()))
