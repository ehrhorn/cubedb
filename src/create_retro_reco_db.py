from datetime import datetime
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from retro_funcs import convert_retro_reco_energy_to_neutrino_energy

from I3Tray import I3Tray
from icecube import icetray
from icecube import dataclasses
from icecube import simclasses
from icecube import recclasses
from icecube import dataio
from icecube import millipede, photonics_service
from icecube.common_variables import time_characteristics

reconstruction_columns = {
    "event_no": {"type": int, "nullable": False, "primary": True},
    "energy_log10": {
        "type": float,
        "nullable": True,
        "primary": False,
    },
    "time": {"type": float, "nullable": True, "primary": False},
    "vertex_x": {"type": float, "nullable": True, "primary": False},
    "vertex_y": {"type": float, "nullable": True, "primary": False},
    "vertex_z": {"type": float, "nullable": True, "primary": False},
    "azimuth": {"type": float, "nullable": True, "primary": False},
    "zenith": {"type": float, "nullable": True, "primary": False},
    "pid": {"type": int, "nullable": True, "primary": False},
    "cascade_energy": {"type": float, "nullable": True, "primary": False},
    "length": {"type": int, "nullable": True, "primary": False},
}


def sql_create_reconstruction_table(table_name):
    query = """
        CREATE TABLE IF NOT EXISTS {} (
            event_no INTEGER PRIMARY KEY NOT NULL,
            energy_log10 REAL,
            time REAL,
            vertex_x REAL,
            vertex_y REAL,
            vertex_z REAL,
            azimuth REAL,
            zenith REAL,
            pid INTEGER,
            cascade_energy FLOAT,
            length FLOAT
        );
    """.format(
        table_name
    )
    return query


def sql_update_reconstruction(table_name):
    query = """
        INSERT INTO {}({}) VALUES ({})
    """.format(
        table_name,
        ", ".join(list(reconstruction_columns.keys())),
        ", ".join(["?"] * len(list(reconstruction_columns.keys()))),
    )
    return query


def create_db(out_db, table_name):
    print("{}: creating DB".format(datetime.now()))
    with sqlite3.connect(str(out_db)) as con:
        cursor = con.cursor()
        cursor.execute(sql_create_reconstruction_table(table_name))


def get_candidate_events(meta_db):
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
            abs(meta.pid) in (12, 14, 16)
            and run_types.run_type in ('genie')
            and reconstruction_names.reconstruction_name in ('retro_crs_prefit__median__neutrino')
        limit -1
    """
    with sqlite3.connect(str(meta_db)) as con:
        candidate_events = pd.read_sql(query, con)
    return candidate_events


def fetch_events(frame, inputs):
    reconstruction = inputs[0]
    dataframe = inputs[1]

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
    pid = dataframe["pid"].values[0]
    reconstruction_name = dataframe["reconstruction_name"].values[0]
    event_length = dataframe["raw_event_length"].values[0]

    pulses = frame["SplitInIcePulses"].apply(frame)

    test_event_length = sum(len(x) for x in pulses.values())

    assert event_length == test_event_length, "Whoops! Event lenghts are not the same!"

    reconstruction_result = frame[reconstruction_name]
    reconstruction_temp = np.zeros(len(reconstruction_columns))

    reconstruction_position = reconstruction_result.pos
    reconstruction_direction = reconstruction_result.dir

    reconstruction_temp[0] = event_no
    reconstruction_temp[2] = reconstruction_result.time
    reconstruction_temp[3] = reconstruction_position.x
    reconstruction_temp[4] = reconstruction_position.y
    reconstruction_temp[5] = reconstruction_position.z
    reconstruction_temp[6] = reconstruction_direction.azimuth
    reconstruction_temp[7] = reconstruction_direction.zenith
    reconstruction_temp[8] = reconstruction_result.pdg_encoding
    reconstruction_temp[9] = frame["retro_crs_prefit__cascade_energy"]["median"]
    reconstruction_temp[10] = frame["retro_crs_prefit__median__track"].length
    reconstruction_temp[np.isinf(reconstruction_temp)] = np.nan

    energy = convert_retro_reco_energy_to_neutrino_energy(
        frame["retro_crs_prefit__cascade_energy"]["median"],
        frame["retro_crs_prefit__median__track"].length,
    )
    reconstruction_temp[1] = np.log10(energy[2])

    reconstruction.append(
        tuple([reconstruction_temp[i] for i in range(reconstruction_temp.shape[0])])
    )


def i3_to_list_of_tuples(inputs):
    i3_file = inputs[0]
    gcd_file = inputs[1]
    dataframe = inputs[2]
    reconstruction = []
    tray = I3Tray()
    tray.AddModule("I3Reader", "reader", FilenameList=[str(gcd_file)] + [str(i3_file)])
    tray.Add(
        fetch_events,
        "fetch_events",
        inputs=(reconstruction, dataframe),
    )
    tray.Execute()
    tray.Finish()
    return reconstruction


meta_db = Path().home().joinpath("work").joinpath("datasets").joinpath("meta.db")
out_db = Path().home().joinpath("data").joinpath("retro_reco_2.db")
table_name = "retro_reco"
create_db(out_db, table_name)

candidate_events = get_candidate_events(meta_db)

deltas = []

for i, files in enumerate(candidate_events["files"].unique()):
    start = datetime.now()
    events = candidate_events[candidate_events["files"] == files]
    i3_file = files.split(",")[0]
    gcd_file = files.split(",")[1]
    print("{}: fetching from {}".format(datetime.now(), Path(i3_file).name))
    reconstruction = i3_to_list_of_tuples((i3_file, gcd_file, events))
    with sqlite3.connect(str(out_db)) as con:
        cur = con.cursor()
        print("{}: inserting {} into DB".format(datetime.now(), Path(i3_file).name))
        cur.executemany(sql_update_reconstruction(table_name), reconstruction)
        con.commit()
    end = datetime.now()
    delta = (end - start).total_seconds()
    deltas.append(delta)
    avg_time = sum(deltas) / len(deltas)
    files_left = len(candidate_events["files"].unique()) - (i + 1)
    eta = avg_time * files_left / 3600
    print(
        "{}: {} files down, {} to go, ETA {} hours".format(
            datetime.now(), i + 1, files_left, round(eta, 2)
        )
    )

print("{}: done".format(datetime.now()))
