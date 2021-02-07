import os
import pickle
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from I3Tray import I3Tray
from icecube import dataclasses, icetray
from icecube.common_variables import time_characteristics

feature_columns = {
    "event_no": {"type": int, "nullable": False, "primary": False},
    "pulse_no": {"type": int, "nullable": False, "primary": False},
    "string": {"type": int, "nullable": False, "primary": False},
    "dom": {"type": int, "nullable": False, "primary": False},
    "om": {"type": int, "nullable": False, "primary": False},
    "x": {"type": float, "nullable": False, "primary": False},
    "y": {"type": float, "nullable": False, "primary": False},
    "z": {"type": float, "nullable": False, "primary": False},
    "time": {"type": float, "nullable": False, "primary": False},
    "charge": {"type": float, "nullable": False, "primary": False},
    "lc": {"type": bool, "nullable": False},
    "atwd": {"type": bool, "nullable": False},
    "fadc": {"type": bool, "nullable": False},
    "pulse_width": {"type": int, "nullable": False},
    "SplitInIcePulses": {"type": bool, "nullable": False, "primary": False},
    "SRTInIcePulses": {"type": bool, "nullable": True, "primary": False},
}
truth_columns = {
    "event_no": {"type": int, "nullable": False, "primary": True},
    "energy_log10": {"type": float, "nullable": True, "primary": False},
    "time": {"type": float, "nullable": True, "primary": False},
    "vertex_x": {"type": float, "nullable": True, "primary": False},
    "vertex_y": {"type": float, "nullable": True, "primary": False},
    "vertex_z": {"type": float, "nullable": True, "primary": False},
    "direction_x": {"type": float, "nullable": True, "primary": False},
    "direction_y": {"type": float, "nullable": True, "primary": False},
    "direction_z": {"type": float, "nullable": True, "primary": False},
    "azimuth": {"type": float, "nullable": True, "primary": False},
    "zenith": {"type": float, "nullable": True, "primary": False},
    "pid": {"type": int, "nullable": True, "primary": False},
}
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
}
meta_columns = {
    "level": {"type": int, "nullable": True, "primary": False},
    "pid": {"type": int, "nullable": True, "primary": False},
    "raw_event_length": {"type": int, "nullable": False, "primary": False},
    "raw_strings": {"type": int, "nullable": False, "primary": False},
    "raw_doms": {"type": int, "nullable": False, "primary": False},
    "raw_oms": {"type": int, "nullable": False, "primary": False},
    "raw_integrated_charge": {"type": int, "nullable": False, "primary": False},
    "clean_event_length": {"type": int, "nullable": False, "primary": False},
    "clean_strings": {"type": int, "nullable": False, "primary": False},
    "clean_doms": {"type": int, "nullable": False, "primary": False},
    "clean_oms": {"type": int, "nullable": False, "primary": False},
    "clean_integrated_charge": {"type": int, "nullable": False, "primary": False},
    "energy_log10": {"type": int, "nullable": True, "primary": False},
    "unix_start_time": {"type": int, "nullable": False, "primary": False},
    "reconstruction": {"type": int, "nullable": True, "primary": False},
    "event_id": {"type": int, "nullable": False, "primary": False},
    "sub_event_id": {"type": int, "nullable": False, "primary": False},
    "file_number": {"type": int, "nullable": False, "primary": False},
    "i3_file_path": {"type": int, "nullable": False, "primary": False},
    "i3_file_name": {"type": int, "nullable": False, "primary": False},
    "gcd_file_path": {"type": int, "nullable": False, "primary": False},
    "gcd_file_name": {"type": int, "nullable": False, "primary": False},
    "run_type": {"type": int, "nullable": False, "primary": False},
    "run_id": {"type": int, "nullable": False, "primary": False},
    "sub_run_id": {"type": int, "nullable": False, "primary": False},
    "max_size_bytes": {"type": int, "nullable": False, "primary": False},
}
sql_create_reconstruction_names_table = """
    CREATE TABLE reconstruction_names (
        row INTEGER PRIMARY KEY NOT NULL,
        reconstruction_name TEXT NOT NULL
    );
"""
sql_create_i3_file_paths_table = """
    CREATE TABLE i3_file_paths (
        row INTEGER PRIMARY KEY NOT NULL,
        i3_file_path TEXT NOT NULL
    );
"""
sql_create_i3_file_names_table = """
    CREATE TABLE i3_file_names (
        row INTEGER PRIMARY KEY NOT NULL,
        i3_file_name TEXT NOT NULL
    );
"""
sql_create_gcd_file_paths_table = """
    CREATE TABLE gcd_file_paths (
        row INTEGER PRIMARY KEY NOT NULL,
        gcd_file_path TEXT NOT NULL
    );
"""
sql_create_gcd_file_names_table = """
    CREATE TABLE gcd_file_names (
        row INTEGER PRIMARY KEY NOT NULL,
        gcd_file_name TEXT NOT NULL
    );
"""
sql_create_run_types_table = """
    CREATE TABLE run_types (
        row INTEGER PRIMARY KEY NOT NULL,
        run_type TEXT NOT NULL
    );
"""
sql_create_meta_table = """
    CREATE TABLE meta (
        event_no INTEGER PRIMARY KEY NOT NULL,
        level INTEGER NOT NULL,
        pid INTEGER NOT NULL,
        raw_event_length INTEGER NOT NULL,
        raw_strings INTEGER NOT NULL,
        raw_doms INTEGER NOT NULL,
        raw_oms INTEGER NOT NULL,
        raw_integrated_charge REAL NOT NULL,
        clean_event_length INTEGER NOT NULL,
        clean_strings INTEGER NOT NULL,
        clean_doms INTEGER NOT NULL,
        clean_oms INTEGER NOT NULL,
        clean_integrated_charge REAL NOT NULL,
        energy_log10 REAL,
        unix_start_time INTEGER NOT NULL,
        reconstruction INTEGER NOT NULL,
        event_id INTEGER NOT NULL,
        sub_event_id INTEGER NOT NULL,
        file_number INTEGER NOT NULL,
        i3_file_path INTEGER NOT NULL,
        i3_file_name INTEGER NOT NULL,
        gcd_file_path INTEGER NOT NULL,
        gcd_file_name INTEGER NOT NULL,
        run_type INTEGER NOT NULL,
        run_id INTEGER NOT NULL,
        sub_run_id INTEGER NOT NULL,
        max_size_bytes INTEGER NOT NULL
    );
"""
sql_insert_meta = """
    INSERT INTO meta({}) VALUES ({})
""".format(
    ", ".join(list(meta_columns.keys())),
    ", ".join(["?"] * len(list(meta_columns.keys()))),
)


def sql_insert_category(table, column, value):
    query = """
        INSERT INTO {}({}) VALUES ('{}')
    """.format(
        table, column, value
    )
    return query


def sql_get_category(table, column, value):
    query = """
        select row from {} where {} = '{}'
    """.format(
        table, column, value
    )
    return query


def sql_update_meta(event_no):
    query = """
        update meta set {} where event_no = {}
    """.format(
        ", ".join(["{} = ?".format(key) for key in meta_columns]), event_no
    )
    return query


def sql_get_distinct_levels(pid, run_type):
    pid = int(pid)
    query = """
        select
            distinct(level)
        from
            meta
        where
            run_type = '{}'
            and abs(pid) = {}
    """.format(
        run_type, pid
    )
    return query


def sql_get_event_no(run_type, pid, file_number, event_id, sub_event_id):
    pid = int(pid)
    query = """
        select
            event_no
        from
            meta
        where
            abs(pid) = {}
            and file_number = '{}'
            and run_type = '{}'
            and event_id = {}
            and sub_event_id = {}
    """.format(
        pid,
        file_number,
        run_type,
        event_id,
        sub_event_id,
    )
    return query


def get_field_size_in_bytes(column_type):
    if column_type == int:
        size = 8
    elif column_type == float:
        size = 8
    elif column_type == bool:
        size = 1
    return size


def check_category_and_insert(db, table, column, value):
    with sqlite3.connect(str(db)) as con:
        query = sql_get_category(table, column, value)
        cur = con.cursor()
        cur.execute(query)
        category = cur.fetchall()
    if not category:
        with sqlite3.connect(str(db)) as con:
            query = sql_insert_category(table, column, value)
            cur = con.cursor()
            cur.execute(query)
            con.commit()
        with sqlite3.connect(str(db)) as con:
            query = sql_get_category(table, column, value)
            cur = con.cursor()
            cur.execute(query)
            category = cur.fetchall()
    category = category[0][0]
    return category


def get_event_size_in_bytes(pid, reconstruction, event_length):
    size = []
    for key in feature_columns:
        size.append(
            event_length * get_field_size_in_bytes(feature_columns[key]["type"])
        )
    if pid > 0:
        for key in truth_columns:
            size.append(get_field_size_in_bytes(truth_columns[key]["type"]))
    if reconstruction != "none":
        for key in reconstruction_columns:
            size.append(get_field_size_in_bytes(reconstruction_columns[key]["type"]))
    size = sum(size)
    return size


def create_db(db_file):
    print("{}: creating DB".format(datetime.now().strftime("%H:%M:%S")))
    with sqlite3.connect(str(db_file)) as con:
        cursor = con.cursor()
        cursor.execute(sql_create_meta_table)
        cursor.execute(sql_create_reconstruction_names_table)
        cursor.execute(sql_create_i3_file_paths_table)
        cursor.execute(sql_create_i3_file_names_table)
        cursor.execute(sql_create_gcd_file_paths_table)
        cursor.execute(sql_create_gcd_file_names_table)
        cursor.execute(sql_create_run_types_table)


def fetch_meta(frame, inputs):
    meta = inputs[0]
    i3_file = inputs[1]
    gcd_file = inputs[2]
    level = inputs[3]
    pid = inputs[4]
    run_type = inputs[5]
    file_number = inputs[6]
    db = inputs[7]
    i3_file_path = inputs[8]
    i3_file_name = inputs[9]
    gcd_file_path = inputs[10]
    gcd_file_name = inputs[11]

    event_id = frame["I3EventHeader"].event_id
    sub_event_id = frame["I3EventHeader"].sub_event_id

    if frame["I3EventHeader"].sub_event_stream != "InIceSplit":
        return False

    if "SRTInIcePulses" not in frame:
        if "SplitInIcePulsesSRT" not in frame:
            return False

    try:
        uncleaned_pulses = frame["SplitInIcePulses"].apply(frame)
        cleaned_pulses = frame["SRTInIcePulses"].apply(frame)
    except Exception as e:
        try:
            uncleaned_pulses = frame["SplitInIcePulses"].apply(frame)
            cleaned_pulses = frame["SplitInIcePulsesSRT"].apply(frame)
        except Exception as e:
            return False

    raw_event_length = sum(len(x) for x in uncleaned_pulses.values())
    clean_event_length = sum(len(x) for x in cleaned_pulses.values())

    try:
        pid = int(pid)
        mc_tree = frame["I3MCTree"]
        if abs(pid) == 13:
            true_primary = dataclasses.get_most_energetic_muon(mc_tree)
        elif abs(pid) == 12 or abs(pid) == 14 or abs(pid) == 16:
            true_primary = dataclasses.get_most_energetic_primary(mc_tree)
    except Exception as e:
        true_primary = None

    if true_primary is None:
        energy_log10 = None
        pid = 0
    else:
        energy_log10 = np.log10(true_primary.energy)
        pid = true_primary.pdg_encoding

    try:
        _ = frame["retro_crs_prefit__median__neutrino"]
        reconstruction = "retro_crs_prefit__median__neutrino"
    except Exception:
        try:
            _ = frame["MonopodFit4"]
            reconstruction = "MonopodFit4"
        except Exception:
            reconstruction = "none"

    event_size = get_event_size_in_bytes(pid, reconstruction, raw_event_length)

    reconstruction = check_category_and_insert(
        db, "reconstruction_names", "reconstruction_name", reconstruction
    )

    dom_geom = frame["I3Geometry"].omgeo

    cleaned_time_list = []
    for om_key, pulses in cleaned_pulses.items():
        for pulse in pulses:
            cleaned_time_list.append(pulse.time)

    raw_strings = []
    raw_doms = []
    raw_oms = []
    raw_integrated_charge = []
    clean_strings = []
    clean_doms = []
    clean_oms = []
    clean_integrated_charge = []

    for om_key, pulses in uncleaned_pulses.items():
        om_geom = dom_geom[om_key]
        for pulse in pulses:
            raw_strings.append(om_key[0])
            raw_doms.append(om_key[1])
            raw_oms.append(om_key[2])
            raw_integrated_charge.append(pulse.charge)
            if pulse.time in cleaned_time_list:
                clean_strings.append(om_key[0])
                clean_doms.append(om_key[1])
                clean_oms.append(om_key[2])
                clean_integrated_charge.append(pulse.charge)

    raw_strings = len(np.unique(raw_strings))
    raw_doms = len(np.unique(raw_doms))
    raw_oms = len(np.unique(raw_oms))
    raw_integrated_charge = sum(raw_integrated_charge)
    clean_strings = len(np.unique(clean_strings))
    clean_doms = len(np.unique(clean_doms))
    clean_oms = len(np.unique(clean_oms))
    clean_integrated_charge = sum(clean_integrated_charge)

    meta.append(
        (
            level,
            pid,
            int(raw_event_length),
            int(raw_strings),
            int(raw_doms),
            int(raw_oms),
            raw_integrated_charge,
            int(clean_event_length),
            int(clean_strings),
            int(clean_doms),
            int(clean_oms),
            clean_integrated_charge,
            energy_log10,
            int(frame["I3EventHeader"].start_time.unix_time),
            reconstruction,
            event_id,
            sub_event_id,
            file_number,
            i3_file_path,
            i3_file_name,
            gcd_file_path,
            gcd_file_name,
            run_type,
            frame["I3EventHeader"].run_id,
            frame["I3EventHeader"].sub_run_id,
            event_size,
        )
    )


def get_meta_from_i3(
    i3_file,
    gcd_file,
    level,
    particle_type,
    run_type,
    file_number,
    db,
    i3_file_path,
    i3_file_name,
    gcd_file_path,
    gcd_file_name,
):
    meta = []
    temp_idxs = []
    tray = I3Tray()
    tray.AddModule("I3Reader", "reader", FilenameList=[str(gcd_file)] + [str(i3_file)])
    tray.Add(
        fetch_meta,
        "fetch_meta",
        inputs=(
            meta,
            Path(i3_file),
            Path(gcd_file),
            level,
            particle_type,
            run_type,
            file_number,
            db,
            i3_file_path,
            i3_file_name,
            gcd_file_path,
            gcd_file_name,
            temp_idxs,
        ),
    )
    tray.Execute()
    tray.Finish()
    return meta


db_file = Path().home().joinpath("meta.db")
raw_files_root = Path().home().joinpath("work").joinpath("datasets")
i3_files_dict = raw_files_root.joinpath("files.pkl")
with open(str(i3_files_dict), "rb") as f:
    i3_files = pickle.load(f)

no_files = 0
for key, value in i3_files.items():
    for pid in value:
        no_files += len(value[pid]["files"])
done_files = 0
time_deltas = []

files = []
if not db_file.is_file():
    create_db(db_file)
else:
    with sqlite3.connect(str(db_file)) as con:
        query = "select distinct(i3_file_name) from i3_file_names"
        cur = con.cursor()
        cur.execute(query)
        files = cur.fetchall()
    files = [result[0] for result in files]
    no_files = no_files - len(files)

for key, value in i3_files.items():
    run_type = key.split("_")[0]
    run_type = check_category_and_insert(db_file, "run_types", "run_type", run_type)
    if "_lvl2" in key:
        level = 2
    elif "_lvl3" in key:
        level = 3
    elif "_lvl4" in key:
        level = 4
    elif "_lvl5" in key:
        level = 5
    elif "_step4" in key:
        level = 2
    else:
        level = None
    for particle_type in value:
        gcd_file = value[particle_type]["gcd"]
        gcd_file_path = check_category_and_insert(
            db_file, "gcd_file_paths", "gcd_file_path", str(Path(gcd_file).parent)
        )
        gcd_file_name = check_category_and_insert(
            db_file, "gcd_file_names", "gcd_file_name", Path(gcd_file).name
        )
        for i, i3_file in enumerate(sorted(value[particle_type]["files"])):
            start = datetime.now()
            if files and Path(i3_file).name in files:
                print(
                    "{}: {} already present in db; skipping".format(
                        datetime.now().strftime("%H:%M:%S"), i3_file
                    )
                )
                continue
            i3_file_path = check_category_and_insert(
                db_file, "i3_file_paths", "i3_file_path", str(Path(i3_file).parent)
            )
            i3_file_name = check_category_and_insert(
                db_file, "i3_file_names", "i3_file_name", Path(i3_file).name
            )
            pattern = r"[0-9]{8}"
            results = re.findall(pattern, Path(i3_file).name)
            if not results:
                pattern = r"[0-9]{6}"
                results = re.findall(pattern, Path(i3_file).name)
            file_number = int(results[-1])
            print(
                "{}: fetching from {}".format(
                    datetime.now().strftime("%H:%M:%S"), Path(i3_file).name
                )
            )
            if particle_type == "data":
                pid = 0
            else:
                pid = particle_type
            meta = get_meta_from_i3(
                i3_file,
                gcd_file,
                level,
                pid,
                run_type,
                file_number,
                db_file,
                i3_file_path,
                i3_file_name,
                gcd_file_path,
                gcd_file_name,
            )
            print(
                "{}: inserting {} into db".format(
                    datetime.now().strftime("%H:%M:%S"), Path(i3_file).name
                )
            )
            with sqlite3.connect(str(db_file)) as con:
                cur = con.cursor()
                cur.executemany(sql_insert_meta, meta)
                con.commit()
            os.remove(i3_file)
            done_files += 1
            end = datetime.now()
            time_delta = (end - start).total_seconds()
            time_deltas.append(time_delta)
            avg_time = sum(time_deltas) / len(time_deltas)
            eta_seconds = avg_time * (no_files - done_files)
            now = datetime.now().strftime("%H:%M:%S")
            print(
                "{}: {} files down, {} to go, around {} hours left".format(
                    now, done_files, no_files - done_files, int(eta_seconds / 3600.0)
                )
            )
            break
        break
    break

now = datetime.now().strftime("%H:%M:%S")
print("{}: Done!".format(now))
