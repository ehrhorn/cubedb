from datetime import datetime
from datetime import timedelta
import os
from pathlib import Path
import pickle
import re
import sqlite3

import numpy as np

from I3Tray import I3Tray
from icecube import icetray
from icecube import dataclasses
from icecube.common_variables import time_characteristics

meta_columns = {
    "level": {"type": int, "nullable": True, "primary": False},
    "pid": {"type": int, "nullable": True, "primary": False},
    "event_length": {"type": int, "nullable": False, "primary": False},
    "strings": {"type": int, "nullable": False, "primary": False},
    "doms": {"type": int, "nullable": False, "primary": False},
    "pmts": {"type": int, "nullable": False, "primary": False},
    "integrated_charge": {"type": int, "nullable": False, "primary": False},
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
        event_length INTEGER NOT NULL,
        strings INTEGER NOT NULL,
        doms INTEGER NOT NULL,
        pmts INTEGER NOT NULL,
        integrated_charge REAL NOT NULL,
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
        sub_run_id INTEGER NOT NULL
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


def create_db(db_file):
    print("{}: creating DB".format(datetime.now()))
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
    level = inputs[1]
    pid = inputs[2]
    run_type = inputs[3]
    file_number = inputs[4]
    db = inputs[5]
    i3_file_path = inputs[6]
    i3_file_name = inputs[7]
    gcd_file_path = inputs[8]
    gcd_file_name = inputs[9]

    event_id = frame["I3EventHeader"].event_id
    sub_event_id = frame["I3EventHeader"].sub_event_id

    if frame["I3EventHeader"].sub_event_stream != "InIceSplit":
        return False

    if "SRTInIcePulses" not in frame:
        if "SplitInIcePulsesSRT" not in frame:
            return False

    pulses = frame["SplitInIcePulses"].apply(frame)

    event_length = sum(len(x) for x in pulses.values())

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

    reconstruction = check_category_and_insert(
        db, "reconstruction_names", "reconstruction_name", reconstruction
    )

    strings = []
    doms = []
    pmts = []
    integrated_charge = []

    for om_key, pulses in pulses.items():
        for pulse in pulses:
            strings.append(om_key[0])
            doms.append(om_key[1])
            pmts.append(om_key[2])
            integrated_charge.append(pulse.charge)

    strings = len(np.unique(strings))
    doms = len(np.unique(doms))
    pmts = len(np.unique(pmts))
    integrated_charge = sum(integrated_charge)

    meta.append(
        (
            level,
            pid,
            int(event_length),
            int(strings),
            int(doms),
            int(pmts),
            integrated_charge,
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


db_file = Path().home().joinpath("data").joinpath("meta_2.db")
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
                        datetime.now(), i3_file
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
            print("{}: fetching from {}".format(datetime.now(), Path(i3_file).name))
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
            print("{}: inserting {} into db".format(datetime.now(), Path(i3_file).name))
            with sqlite3.connect(str(db_file)) as con:
                cur = con.cursor()
                cur.executemany(sql_insert_meta, meta)
                con.commit()
            done_files += 1
            end = datetime.now()
            time_delta = (end - start).total_seconds()
            time_deltas.append(time_delta)
            avg_time = sum(time_deltas) / len(time_deltas)
            files_left = no_files - done_files
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

print("{}: Done!".format(datetime.now()))
