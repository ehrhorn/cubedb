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
    "event_no": {"type": int, "nullable": False, "primary": False},
    "string": {"type": int, "nullable": False, "primary": False},
    "dom": {"type": int, "nullable": False, "primary": False},
    "pmt": {"type": int, "nullable": False, "primary": False},
    "dom_x": {"type": float, "nullable": False, "primary": False},
    "dom_y": {"type": float, "nullable": False, "primary": False},
    "dom_z": {"type": float, "nullable": False, "primary": False},
    "pmt_x": {"type": float, "nullable": False, "primary": False},
    "pmt_y": {"type": float, "nullable": False, "primary": False},
    "pmt_z": {"type": float, "nullable": False, "primary": False},
    "pmt_area": {"type": float, "nullable": False, "primary": False},
    "pmt_type": {"type": int, "nullable": False, "primary": False},
    "time": {"type": float, "nullable": False, "primary": False},
    "charge_log10": {"type": float, "nullable": False, "primary": False},
    "lc": {"type": bool, "nullable": False},
    "pulse_width": {"type": int, "nullable": False},
    "SplitInIcePulses": {"type": bool, "nullable": False, "primary": False},
    "SRTInIcePulses": {"type": bool, "nullable": True, "primary": False},
}
truth_columns = {
    "event_no": {"type": int, "nullable": False, "primary": True},
    "energy_log10": {"type": float, "nullable": True, "primary": False},
    "time": {"type": float, "nullable": True, "primary": False},
    "position_x": {"type": float, "nullable": True, "primary": False},
    "position_y": {"type": float, "nullable": True, "primary": False},
    "position_z": {"type": float, "nullable": True, "primary": False},
    "direction_x": {"type": float, "nullable": True, "primary": False},
    "direction_y": {"type": float, "nullable": True, "primary": False},
    "direction_z": {"type": float, "nullable": True, "primary": False},
    "azimuth": {"type": float, "nullable": True, "primary": False},
    "zenith": {"type": float, "nullable": True, "primary": False},
    "pid": {"type": int, "nullable": True, "primary": False},
    "interaction_type": {"type": int, "nullable": True, "primary": False},
    "muon_track_length": {"type": float, "nullable": True, "primary": False},
    "stopped_muon": {"type": int, "nullable": True, "primary": False},
}

sql_create_features_table = """
    CREATE TABLE features (
        row INTEGER PRIMARY KEY NOT NULL,
        event_no INTEGER NOT NULL,
        string INTEGER NOT NULL,
        dom INTEGER NOT NULL,
        pmt INTEGER NOT NULL,
        dom_x REAL NOT NULL,
        dom_y REAL NOT NULL,
        dom_z REAL NOT NULL,
        pmt_x REAL NOT NULL,
        pmt_y REAL NOT NULL,
        pmt_z REAL NOT NULL,
        pmt_area REAL NOT NULL,
        pmt_type INTEGER NOT NULL,
        time INTEGER NOT NULL,
        charge_log10 REAL NOT NULL,
        lc INTEGER,
        pulse_width INTEGER,
        SplitInIcePulses INTEGER,
        SRTInIcePulses INTEGER
    );
"""

sql_create_truth_table = """
    CREATE TABLE IF NOT EXISTS truth (
        event_no INTEGER PRIMARY KEY NOT NULL,
        energy_log10 REAL,
        time REAL,
        position_x REAL,
        position_y REAL,
        position_z REAL,
        direction_x REAL,
        direction_y REAL,
        direction_z REAL,
        azimuth REAL,
        zenith REAL,
        pid INTEGER,
        interaction_type INTEGER,
        muon_track_length REAL,
        stopped_muon INTEGER
    );
"""

sql_update_features = """
    INSERT INTO features({}) VALUES ({})
""".format(
    ", ".join(list(feature_columns.keys())),
    ", ".join(["?"] * len(list(feature_columns.keys()))),
)
sql_update_truth = """
    INSERT INTO truth({}) VALUES ({})
""".format(
    ", ".join(list(truth_columns.keys())),
    ", ".join(["?"] * len(list(truth_columns.keys()))),
)


def create_db(out_db):
    print("{}: creating DB".format(datetime.now()))
    with sqlite3.connect(str(out_db)) as con:
        cursor = con.cursor()
        cursor.execute(sql_create_features_table)


def create_truth_table(out_db):
    print("{}: creating truth table".format(datetime.now()))
    with sqlite3.connect(str(out_db)) as con:
        cursor = con.cursor()
        cursor.execute(sql_create_truth_table)


def get_candidate_events(meta_db, query):
    with sqlite3.connect(str(meta_db)) as con:
        candidate_events = pd.read_sql(query, con)
    return candidate_events


def calculate_displaced_point(position, direction, length):
    position_to_direction = direction - position
    normed_position_to_direction = position_to_direction / np.linalg.norm(
        position_to_direction
    )
    displaced_point = position + length * normed_position_to_direction
    return displaced_point


def check_if_point_inside_cylinder(position, direction, length):
    displaced_point = calculate_displaced_point(position, direction, length)
    origin, z, radius = fiducial_volume_icecube()
    pt1 = np.array((origin[0], origin[1], z[0]))
    pt2 = np.array((origin[0], origin[1], z[1]))
    vec = pt2 - pt1
    const = radius * np.linalg.norm(vec)
    inside_ends = (
        np.dot(displaced_point - pt1, vec) >= 0
        and np.dot(displaced_point - pt2, vec) <= 0
    )
    inside_radius = np.linalg.norm(np.cross(displaced_point - pt1, vec)) <= const
    if inside_ends and inside_radius:
        return 1
    else:
        return 0


def fiducial_volume_deepcore():
    origin = np.array((46.29, -34.88, -330.0))
    z = np.array((-500, -200))
    radius = 150
    return origin, z, radius


def fiducial_volume_icecube():
    origin = np.array((0, 0, 0))
    z = np.array((-513, 525))
    radius = 600
    return origin, z, radius


def fetch_events(frame, inputs):
    features = inputs[0]
    truth = inputs[1]
    dataframe = inputs[2]

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
    event_length = dataframe["raw_event_length"].values[0]

    try:
        uncleaned_pulses = frame["SplitInIcePulses"].apply(frame)
        cleaned_pulses = frame["SRTInIcePulses"].apply(frame)
    except Exception as e:
        try:
            uncleaned_pulses = frame["SplitInIcePulses"].apply(frame)
            cleaned_pulses = frame["SplitInIcePulsesSRT"].apply(frame)
        except Exception as e:
            return False

    test_event_length = sum(len(x) for x in uncleaned_pulses.values())

    assert event_length == test_event_length, "Whoops! Event lenghts are not the same!"

    try:
        mc_tree = frame["I3MCTree"]
        truth_valid = True
        if abs(pid) == 13:
            true_primary = dataclasses.get_most_energetic_muon(mc_tree)
        else:
            true_primary = dataclasses.get_most_energetic_primary(mc_tree)
    except Exception as e:
        truth_valid = False

    dom_geom = frame["I3Geometry"].omgeo

    features_temp = np.zeros((event_length, len(feature_columns) - 1))
    truth_temp = np.zeros(len(truth_columns))

    cleaned_time_list = []
    for om_key, pulses in cleaned_pulses.items():
        for pulse in pulses:
            cleaned_time_list.append(pulse.time)
    row = 0
    for om_key, pulses in uncleaned_pulses.items():
        om_geom = dom_geom[om_key]
        om_position = om_geom.position
        om_orientation = om_geom.orientation
        om_area = om_geom.area
        om_type = om_geom.omtype
        for pulse in pulses:
            features_temp[row, :] = [
                om_key[0],
                om_key[1],
                om_key[2],
                om_position.x,
                om_position.y,
                om_position.z,
                om_orientation.x,
                om_orientation.y,
                om_orientation.z,
                om_area,
                om_type,
                pulse.time,
                pulse.charge,
                (pulse.flags & 0x1) >> 0,
                pulse.width,
                1,
                1 if pulse.time in cleaned_time_list else 0,
            ]
            row += 1
    features_temp = features_temp[features_temp[:, 10].argsort()]
    for i in range(features_temp.shape[0]):
        features.append(
            (
                int(event_no),
                int(features_temp[i, 0]),
                int(features_temp[i, 1]),
                int(features_temp[i, 2]),
                float(features_temp[i, 3]),
                float(features_temp[i, 4]),
                float(features_temp[i, 5]),
                float(features_temp[i, 6]),
                float(features_temp[i, 7]),
                float(features_temp[i, 8]),
                float(features_temp[i, 9]),
                float(features_temp[i, 10]),
                float(features_temp[i, 11]),
                float(np.log10(features_temp[i, 12])),
                int(features_temp[i, 13]),
                int(features_temp[i, 14]),
                int(features_temp[i, 15]),
                int(features_temp[i, 16]),
            )
        )

    if truth_valid:
        true_primary_direction = true_primary.dir
        true_primary_entry_position = true_primary.pos
        truth_temp[0] = event_no
        truth_temp[1] = np.log10(true_primary.energy)
        truth_temp[2] = true_primary.time
        truth_temp[3] = true_primary_entry_position.x
        truth_temp[4] = true_primary_entry_position.y
        truth_temp[5] = true_primary_entry_position.z
        truth_temp[6] = true_primary_direction.x
        truth_temp[7] = true_primary_direction.y
        truth_temp[8] = true_primary_direction.z
        truth_temp[9] = true_primary_direction.azimuth
        truth_temp[10] = true_primary_direction.zenith
        truth_temp[11] = true_primary.pdg_encoding
        truth_temp[13] = true_primary.length
        position = np.array(
            (
                true_primary_entry_position.x,
                true_primary_entry_position.y,
                true_primary_entry_position.z,
            )
        )
        direction = np.array(
            (
                true_primary_direction.x,
                true_primary_direction.y,
                true_primary_direction.z,
            )
        )
        try:
            truth_temp[12] = frame["I3MCWeightDict"]["InteractionType"]
        except Exception as e:
            truth_temp[12] = np.nan
        if (truth_temp[12] == 1 and abs(pid) == 14) or abs(pid) == 13:
            true_muon = dataclasses.get_most_energetic_muon(mc_tree)
            truth_temp[13] = true_muon.length
            stopped = check_if_point_inside_cylinder(
                position, direction, truth_temp[13]
            )
            truth_temp[14] = stopped
        else:
            truth_temp[13] = np.nan
            truth_temp[14] = np.nan
        truth.append(tuple([truth_temp[i] for i in range(truth_temp.shape[0])]))


def i3_to_list_of_tuples(inputs):
    i3_file = inputs[0]
    gcd_file = inputs[1]
    dataframe = inputs[2]
    features = []
    truth = []
    tray = I3Tray()
    tray.AddModule("I3Reader", "reader", FilenameList=[str(gcd_file)] + [str(i3_file)])
    tray.Add(
        fetch_events,
        "fetch_events",
        inputs=(features, truth, dataframe),
    )
    tray.Execute()
    tray.Finish()
    return features, truth


def create_sqlite_db(dataset_name, query):
    meta_db = Path().home().joinpath("work").joinpath("datasets").joinpath("meta.db")
    data_db = Path().home().joinpath("data").joinpath(dataset_name + ".db")
    create_db(data_db)

    candidate_events = get_candidate_events(meta_db, query)

    deltas = []

    for i, files in enumerate(candidate_events["files"].unique()):
        start = datetime.now()
        events = candidate_events[candidate_events["files"] == files]
        i3_file = files.split(",")[0]
        gcd_file = files.split(",")[1]
        print("{}: fetching from {}".format(datetime.now(), Path(i3_file).name))
        features, truth = i3_to_list_of_tuples((i3_file, gcd_file, events))
        with sqlite3.connect(str(data_db)) as con:
            cur = con.cursor()
            print("{}: inserting {} into DB".format(datetime.now(), Path(i3_file).name))
            cur.executemany(sql_update_features, features)
            con.commit()
        if truth:
            try:
                with sqlite3.connect(str(data_db)) as con:
                    cur = con.cursor()
                    cur.executemany(sql_update_truth, truth)
            except Exception:
                create_truth_table(data_db)
                with sqlite3.connect(str(data_db)) as con:
                    cur = con.cursor()
                    cur.executemany(sql_update_truth, truth)
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
                datetime.now(),
                i + 1,
                files_left,
                int(eta_hour),
                int(eta_min),
                int(eta_sec),
                eta_time,
            )
        )

    print("{}: done".format(datetime.now()))
