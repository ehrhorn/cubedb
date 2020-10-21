from datetime import datetime
from datetime import timedelta
from pathlib import Path
import pickle
import sqlite3

retro_file = Path().home().joinpath("data").joinpath("retro_reco_2.db")
meta_file = Path().home().joinpath("data").joinpath("local_meta.db")
event_no_list = []
output_file = Path().home() / "duplicates.pkl"

try:
    query = "create index duplicate_idx on meta(event_id, sub_event_id, file_number)"
    with sqlite3.connect(meta_file) as con:
        cur = con.cursor()
        cur.execute(query)
except Exception:
    print("Index already exists")

query = "select event_no from retro_reco limit -1"
with sqlite3.connect(retro_file) as con:
    cur = con.cursor()
    cur.execute(query)
    retro_ids = cur.fetchall()

done_events = 0
time_deltas = []
no_events = len(retro_ids)

for i, retro_id in enumerate(retro_ids):
    start = datetime.now()
    query = """
select
    event_id,
    sub_event_id,
    file_number,
    raw_event_length,
    raw_integrated_charge,
    clean_event_length,
    clean_integrated_charge,
    energy_log10
from
    meta
where
    event_no = {}
    """.format(
        retro_id[0]
    )
    with sqlite3.connect(meta_file) as con:
        cur = con.cursor()
        cur.execute(query)
        event_info = cur.fetchall()
    query = """
select
    event_no,
    raw_event_length,
    raw_integrated_charge,
    clean_event_length,
    clean_integrated_charge,
    energy_log10
from
    meta
where
    event_id = {}
    and sub_event_id = {}
    and file_number = {}
    and event_no is not {}
    """.format(
        event_info[0][0], event_info[0][1], event_info[0][2], retro_id[0]
    )
    with sqlite3.connect(meta_file) as con:
        cur = con.cursor()
        cur.execute(query)
        duplicates = cur.fetchall()
    if len(duplicates) > 0:
        for duplicate in duplicates:
            if (
                duplicate[1] == event_info[0][3]
                and duplicate[2] == event_info[0][4]
                and duplicate[3] == event_info[0][5]
                and duplicate[4] == event_info[0][6]
                and duplicate[5] == event_info[0][7]
            ):
                event_no_list.append(duplicate[0])
    done_events += 1
    end = datetime.now()
    time_delta = (end - start).total_seconds()
    time_deltas.append(time_delta)
    avg_time = sum(time_deltas) / len(time_deltas)
    files_left = no_events - done_events
    eta_seconds = avg_time * files_left
    eta_min, eta_sec = divmod(eta_seconds, 60)
    eta_hour, eta_min = divmod(eta_min, 60)
    eta_time = datetime.now() + timedelta(seconds=eta_seconds)
    if i % 1000 == 0 and i > 0:
        print(
            "{}: {} events down, {} to go, ETA {:02d}:{:02d}:{:02d}; that's {}".format(
                datetime.now(),
                i + 1,
                files_left,
                int(eta_hour),
                int(eta_min),
                int(eta_sec),
                eta_time,
            )
        )

with open(output_file, "wb") as f:
    pickle.dump(event_no_list, f)

print("{}: Done.".format(datetime.now()))
