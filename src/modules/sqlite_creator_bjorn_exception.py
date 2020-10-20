from datetime import datetime
from datetime import timedelta
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from I3Tray import I3Tray
from icecube import icetray
from icecube import dataclasses
from icecube import simclasses
from icecube import recclasses
from icecube import dataio
from icecube import millipede, photonics_service
from icecube.common_variables import time_characteristics

feature_columns = {
    "event_no": {"type": int, "nullable": False, "primary": True},
    "true_primary_energy": {"type": float, "nullable": False, "primary": True},
    "true_primary_time": {"type": float, "nullable": False, "primary": True},
    "true_primary_position_x": {"type": float, "nullable": False, "primary": True},
    "true_primary_position_y": {"type": float, "nullable": False, "primary": True},
    "true_primary_position_z": {"type": float, "nullable": False, "primary": True},
    "true_primary_direction_x": {"type": float, "nullable": False, "primary": True},
    "true_primary_direction_y": {"type": float, "nullable": False, "primary": True},
    "true_primary_direction_z": {"type": float, "nullable": False, "primary": True},
    "interaction_type": {"type": int, "nullable": True, "primary": False},
    "cascade_energy": {"type": float, "nullable": True, "primary": False},
    "length": {"type": int, "nullable": True, "primary": False},
    "file": {"type": str, "nullable": True, "primary": False},
    "idx": {"type": int, "nullable": True, "primary": False},
    "old_idx": {"type": int, "nullable": True, "primary": False},
}

sql_create_features_table = """
    CREATE TABLE features (
        event_no INTEGER PRIMARY KEY NOT NULL,
        true_primary_energy FLOAT,
        true_primary_time INTEGER,
        true_primary_position_x FLOAT,
        true_primary_position_y FLOAT,
        true_primary_position_z FLOAT,
        true_primary_direction_x FLOAT,
        true_primary_direction_y FLOAT,
        true_primary_direction_z FLOAT,
        interaction_type INTEGER,
        cascade_energy FLOAT,
        length FLOAT,
        file TEXT,
        idx INTEGER,
        old_idx INTEGER
    );
"""
sql_update_features = """
    INSERT INTO features({}) VALUES ({})
""".format(
    ", ".join(list(feature_columns.keys())),
    ", ".join(["?"] * len(list(feature_columns.keys()))),
)


def create_db(out_db):
    print("{}: creating DB".format(datetime.now()))
    with sqlite3.connect(str(out_db)) as con:
        cursor = con.cursor()
        cursor.execute(sql_create_features_table)


def get_candidate_events(meta_db, pids, run_types, limit, reconstruction_name):
    if not isinstance(pids, list):
        pids = [pids]
    if not isinstance(run_types, list):
        run_types = [run_types]
    query = """
        select
            meta.event_no,
            meta.level,
            meta.pid,
            meta.unix_start_time,
            meta.energy_log10,
            meta.max_size_bytes,
            meta.raw_event_length,
            meta.event_id,
            meta.sub_event_id,
            reconstruction_names.reconstruction_name,
            i3_file_paths.i3_file_path || '/' || i3_file_names.i3_file_name || ',' || gcd_file_paths.gcd_file_path || '/' || gcd_file_names.gcd_file_name as files,
            run_types.run_type
        from
            meta
            inner join reconstruction_names on meta.reconstruction = reconstruction_names.row
            inner join i3_file_paths on meta.i3_file_path = i3_file_paths.row
            inner join i3_file_names on meta.i3_file_name = i3_file_names.row
            inner join gcd_file_paths on meta.gcd_file_path = gcd_file_paths.row
            inner join gcd_file_names on meta.gcd_file_name = gcd_file_names.row
            inner join run_types on meta.run_type = run_types.row
        where
            abs(meta.pid) in ({pid})
            and run_types.run_type in ('{run_type}')
            and reconstruction_names.reconstruction_name in ('{reconstruction_name}')
        limit {limit}
    """.format(
        pid=", ".join([str(pid) for pid in pids]),
        run_type=", ".join([str(run_type) for run_type in run_types]),
        limit=limit,
        reconstruction_name=reconstruction_name,
    )
    with sqlite3.connect(str(meta_db)) as con:
        candidate_events = pd.read_sql(query, con)
    return candidate_events


def fetch_events(frame, inputs):
    features = inputs[0]
    dataframe = inputs[1]
    counter = inputs[2]

    if frame["I3EventHeader"].sub_event_stream != "InIceSplit":
        return False

    event_id = frame["I3EventHeader"].event_id
    sub_event_id = frame["I3EventHeader"].sub_event_id

    dataframe = dataframe[
        (dataframe["event_id"] == event_id)
        & (dataframe["sub_event_id"] == sub_event_id)
    ]
    if dataframe.empty:
        return False

    event_no = dataframe["event_no"].values[0]

    counter.append(1)

    mc_tree = frame["I3MCTree"]
    true_primary = dataclasses.get_most_energetic_primary(mc_tree)
    true_primary_direction = true_primary.dir
    true_primary_entry_position = true_primary.pos

    features_temp = []
    features_temp.append(int(event_no))
    features_temp.append(float(np.log10(true_primary.energy)))
    features_temp.append(int(true_primary.time))
    features_temp.append(float(true_primary_entry_position.x))
    features_temp.append(float(true_primary_entry_position.y))
    features_temp.append(float(true_primary_entry_position.z))
    features_temp.append(float(true_primary_direction.x))
    features_temp.append(float(true_primary_direction.y))
    features_temp.append(float(true_primary_direction.z))
    features_temp.append(int(frame["I3MCWeightDict"]["InteractionType"]))
    features_temp.append(frame["retro_crs_prefit__cascade_energy"]["median"])
    features_temp.append(frame["retro_crs_prefit__median__track"].length)
    features_temp.append(
        dataframe.files.values[0].split(",")[0].split("/")[-1].replace("i3.zst", "h5")
    )
    features_temp.append(int(dataframe.event_id.values[0]))
    features_temp.append(int(sum(counter) - 1))
    features.append(tuple([features_temp[i] for i in range(len(features_temp))]))


def i3_to_list_of_tuples(inputs):
    i3_file = inputs[0]
    gcd_file = inputs[1]
    dataframe = inputs[2]
    features = []
    counter = []
    tray = I3Tray()
    tray.AddModule("I3Reader", "reader", FilenameList=[str(gcd_file)] + [str(i3_file)])
    tray.Add(
        fetch_events, "fetch_events", inputs=(features, dataframe, counter),
    )
    tray.Execute()
    tray.Finish()
    return features


meta_db = Path().home().joinpath("meta.db")
out_db = Path().home().joinpath("epsilon_bjorn_2.db")
create_db(out_db)

print("{}: getting candidate events".format(datetime.now()))

candidate_events = get_candidate_events(
    meta_db, [12, 14, 16], "genie", -1, "retro_crs_prefit__median__neutrino"
)
order_of_magnitude = (
    int(np.floor(np.log10(candidate_events["max_size_bytes"].sum()))) - 9
)
print(
    "{}: size will maximally be on the order of 1e{} gigabytes".format(
        datetime.now(), order_of_magnitude
    )
)

deltas = []

for i, files in enumerate(candidate_events["files"].unique()):
    start = datetime.now()
    events = candidate_events[candidate_events["files"] == files]
    i3_file = files.split(",")[0]
    gcd_file = files.split(",")[1]
    print("{}: fetching from {}".format(datetime.now(), Path(i3_file).name))
    features = i3_to_list_of_tuples((i3_file, gcd_file, events))
    with sqlite3.connect(str(out_db)) as con:
        cur = con.cursor()
        print("{}: inserting {} into DB".format(datetime.now(), Path(i3_file).name))
        cur.executemany(sql_update_features, features)
        con.commit()
    end = datetime.now()
    delta = (end - start).total_seconds()
    deltas.append(delta)
    avg_time = sum(deltas) / len(deltas)
    files_left = len(candidate_events["files"].unique()) - (i + 1)
    eta_seconds = avg_time * files_left
    eta_min, eta_sec = divmod(eta_seconds, 60)
    eta_hour, eta_min = divmod(eta_min, 60)
    eta_time = datetime.now() + timedelta(seconds=eta_seconds)
    print(
        "{}: {} files down, {} to go, ETA {:02d}:{:02d}:{:02d}; that's {}".format(
            datetime.now(), i + 1, files_left, int(eta_hour), int(eta_min), int(eta_sec), eta_time
        )
    )

print("{}: done".format(datetime.now()))
