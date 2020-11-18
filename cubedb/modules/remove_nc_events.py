import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


def nc_event_remover(dataset_name):
    db_path = Path().home() / "data" / (dataset_name + ".db")

    query = "select event_no from truth where interaction_type = 2"
    with sqlite3.connect(str(db_path)) as con:
        event_nos = pd.read_sql(query, con)

    print(f"{datetime.now()}: {len(event_nos)} events to be removed")

    stringified_event_nos = ", ".join(
        [str(event_no) for event_no in event_nos["event_no"].values.tolist()]
    )
    query = f"delete from truth where event_no in ({stringified_event_nos})"
    with sqlite3.connect(str(db_path)) as con:
        cur = con.cursor()
        cur.execute(query)

    query = f"delete from features where event_no in ({stringified_event_nos})"
    with sqlite3.connect(str(db_path)) as con:
        cur = con.cursor()
        cur.execute(query)
