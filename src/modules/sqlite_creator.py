from datetime import datetime
import sqlite3

import numpy as np

from I3Tray import I3Tray
from icecube import icetray
from icecube import dataclasses
from icecube import simclasses
from icecube import recclasses
from icecube import dataio
from icecube import millipede, photonics_service
from icecube.common_variables import time_characteristics


def create_db(DB, particle_type, level, include_truth, include_reconstruction):
    print("{}: creating DB".format(datetime.now().strftime("%H:%M:%S")))
    with sqlite3.connect(str(DB)) as con:
        cursor = con.cursor()
        cursor.execute(sql_create_features_table)
        if include_truth:
            cursor.execute(sql_create_truth_table)
        if include_reconstruction:
            cursor.execute(sql_create_reconstruction_table)
        cursor.execute(sql_create_meta_table)


def calculate_sigma(data):
    sigma_pos = data["upper_bound"] - data["median"]
    sigma_neg = data["median"] - data["lower_bound"]
    return (sigma_pos + sigma_neg) / 2


def convert_spherical_to_cartesian(zenith, azimuth):
    """Convert spherical coordinates to Cartesian coordinates.

    Assumes unit length.

    Zenith: theta
    Azimuth: phi

    Args:
        zenith (numpy.ndarray): zenith/polar angle
        azimuth (numpy.ndarray): azimuthal angle

    Returns:
        numpy.ndarray: x, y, z (event, coordinates) vector
    """
    theta = zenith
    phi = azimuth
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    vectors = np.array((x, y, z)).T
    return vectors


def fetch_events(frame, inputs):
    features = inputs[0]
    truth = inputs[1]
    reconstruction = inputs[2]
    meta = inputs[3]
    file_name = inputs[4]
    level = inputs[5]
    last_event_no = inputs[6]
    particle_type = inputs[7]

    try:
        uncleaned_pulses = frame["SplitInIcePulses"].apply(frame)
        cleaned_pulses = frame["SRTInIcePulses"].apply(frame)
    except Exception as e:
        try:
            uncleaned_pulses = frame["SplitInIcePulses"].apply(frame)
            cleaned_pulses = frame["SplitInIcePulsesSRT"].apply(frame)
        except Exception as e:
            print(str(e))
            return False

    if particle_type == "neutrino" and level == 5:
        reconstruction_result = frame["retro_crs_prefit__median__neutrino"]
        reconstruction_valid = True
    # if particle_type == "neutrino" and level is None:
    #     reconstruction_result = frame["MonopodFit4"]
    #     reconstruction_valid = True
    else:
        reconstruction_valid = False

    try:
        mc_tree = frame["I3MCTree"]
        truth_valid = True
        if particle_type == "muon":
            true_primary = dataclasses.get_most_energetic_muon(mc_tree)
        elif particle_type == "neutrino" or particle_type == "upgrade":
            true_primary = dataclasses.get_most_energetic_primary(mc_tree)
    except Exception as e:
        truth_valid = False

    dom_geom = frame["I3Geometry"].omgeo

    event_length = sum(len(x) for x in uncleaned_pulses.values())
    event_no = last_event_no[-1] + 1
    last_event_no.append(event_no)

    features_temp = np.zeros((event_length, 14))
    truth_temp = np.zeros(len(truth_columns))
    reconstruction_temp = np.zeros(len(reconstruction_columns))

    cleaned_time_list = []
    for om_key, pulses in cleaned_pulses.items():
        for pulse in pulses:
            cleaned_time_list.append(pulse.time)
    row = 0
    for om_key, pulses in uncleaned_pulses.items():
        om_geom = dom_geom[om_key]
        om_position = om_geom.position
        for pulse in pulses:
            features_temp[row, :] = [
                om_key[0],
                om_key[1],
                om_key[2],
                om_position.x,
                om_position.y,
                om_position.z,
                pulse.time,
                pulse.charge,
                (pulse.flags & 0x1) >> 0,
                (pulse.flags & 0x2) >> 1,
                (pulse.flags & 0x4) >> 2,
                pulse.width,
                1,
                1 if pulse.time in cleaned_time_list else 0,
            ]
            row += 1
    features_temp = features_temp[features_temp[:, 6].argsort()]
    for i in range(features_temp.shape[0]):
        features.append(
            (
                int(event_no),
                int(i),
                int(features_temp[i, 0]),
                int(features_temp[i, 1]),
                int(features_temp[i, 2]),
                float(features_temp[i, 3]),
                float(features_temp[i, 4]),
                float(features_temp[i, 5]),
                float(features_temp[i, 6]),
                float(np.log10(features_temp[i, 7])),
                int(features_temp[i, 8]),
                int(features_temp[i, 9]),
                int(features_temp[i, 10]),
                int(features_temp[i, 11]),
                bool(features_temp[i, 12]),
                bool(features_temp[i, 13]),
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
    else:
        truth_temp = np.full(len(truth_columns), np.nan)
    truth.append(tuple([truth_temp[i] for i in range(truth_temp.shape[0])]))

    if reconstruction_valid:
        reconstruction_position = reconstruction_result.pos
        reconstruction_direction = reconstruction_result.dir
        reconstruction_temp[0] = event_no
        reconstruction_temp[1] = np.log10(reconstruction_result.energy)
        reconstruction_temp[2] = reconstruction_result.time
        reconstruction_temp[3] = reconstruction_position.x
        reconstruction_temp[4] = reconstruction_position.y
        reconstruction_temp[5] = reconstruction_position.z
        reconstruction_temp[9] = reconstruction_direction.azimuth
        reconstruction_temp[10] = reconstruction_direction.zenith
        directions = spherical_to_cartesian(
            reconstruction_temp[9], reconstruction_temp[10]
        )
        reconstruction_temp[6] = directions[0]
        reconstruction_temp[7] = directions[0]
        reconstruction_temp[8] = directions[0]
        reconstruction_temp[11] = reconstruction_result.pdg_encoding
        # try:
        #     reconstruction_temp[9] = abs(
        #         np.log10((calculate_sigma(frame["retro_crs_prefit__energy"])))
        #     )
        #     reconstruction_temp[10] = abs(
        #         calculate_sigma(frame["retro_crs_prefit__time"])
        #     )
        #     reconstruction_temp[11] = abs(calculate_sigma(frame["retro_crs_prefit__x"]))
        #     reconstruction_temp[12] = abs(calculate_sigma(frame["retro_crs_prefit__y"]))
        #     reconstruction_temp[13] = abs(calculate_sigma(frame["retro_crs_prefit__z"]))
        #     reconstruction_temp[14] = abs(
        #         calculate_sigma(frame["retro_crs_prefit__azimuth"])
        #     )
        #     reconstruction_temp[15] = abs(
        #         calculate_sigma(frame["retro_crs_prefit__zenith"])
        #     )
        #     for i in range(9, 16):
        #         if reconstruction_temp[i] == 0.0:
        #             reconstruction_temp[i] = np.nan
        # except Exception as e:
        #     print(str(e))
        #     for i in range(9, 16):
        #         reconstruction_temp[i] = np.nan
        reconstruction_temp[np.isinf(reconstruction_temp)] = np.nan
    else:
        reconstruction_temp = np.full(len(reconstruction_columns), np.nan)
    reconstruction.append(
        tuple([reconstruction_temp[i] for i in range(reconstruction_temp.shape[0])])
    )

    meta.append(
        (
            int(event_no),
            str(file_name),
            int(frame["I3EventHeader"].event_id),
            level,
            str(frame["I3EventHeader"].start_time),
            str(frame["I3EventHeader"].end_time),
        )
    )


def i3_to_list_of_tuples(inputs):
    i3_file = inputs[0]
    gcd_file = inputs[1]
    features = []
    truth = []
    reconstruction = []
    meta = []
    file_name = i3_file.name
    level = inputs[2]
    last_event_no = inputs[3]
    particle_type = inputs[4]
    tray = I3Tray()
    tray.AddModule("I3Reader", "reader", FilenameList=[str(gcd_file)] + [str(i3_file)])
    tray.Add(
        fetch_events,
        "fetch_events",
        inputs=(
            features,
            truth,
            reconstruction,
            meta,
            file_name,
            level,
            last_event_no,
            particle_type,
        ),
    )
    tray.Execute()
    tray.Finish()
    return features, truth, reconstruction, meta, last_event_no


feature_columns = {
    "event_no": {"type": int, "nullable": False, "primary": False},
    "pulse_no": {"type": int, "nullable": False, "primary": False},
    "dom_string": {"type": int, "nullable": False, "primary": False},
    "dom_pmt": {"type": int, "nullable": False, "primary": False},
    "dom_om": {"type": int, "nullable": False, "primary": False},
    "dom_x": {"type": float, "nullable": False, "primary": False},
    "dom_y": {"type": float, "nullable": False, "primary": False},
    "dom_z": {"type": float, "nullable": False, "primary": False},
    "dom_time": {"type": float, "nullable": False, "primary": False},
    "dom_charge": {"type": float, "nullable": False, "primary": False},
    "dom_lc": {"type": bool, "nullable": False},
    "dom_atwd": {"type": bool, "nullable": False},
    "dom_fadc": {"type": bool, "nullable": False},
    "dom_pulse_width": {"type": int, "nullable": False},
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
}
reconstruction_columns = {
    "event_no": {"type": int, "nullable": False, "primary": True},
    "energy_log10": {
        "type": float,
        "nullable": True,
        "primary": False,
    },
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
    # "sigma_energy_log10": {"type": float, "nullable": True, "primary": False},
    # "sigma_time": {"type": float, "nullable": True, "primary": False},
    # "sigma_position_x": {"type": float, "nullable": True, "primary": False},
    # "sigma_position_y": {"type": float, "nullable": True, "primary": False},
    # "sigma_position_z": {"type": float, "nullable": True, "primary": False},
    # "sigma_azimuth": {"type": float, "nullable": True, "primary": False},
    # "sigma_zenith": {"type": float, "nullable": True, "primary": False},
}
meta_columns = {
    "event_no": {"type": int, "nullable": False, "primary": True},
    "file": {"type": str, "nullable": False, "primary": False},
    "idx": {"type": int, "nullable": False, "primary": False},
    "level": {"type": int, "nullable": True, "primary": False},
    "start_time": {"type": str, "nullable": True, "primary": False},
    "end_time": {"type": str, "nullable": True, "primary": False},
}

sql_create_features_table = """
    CREATE TABLE features (
        row INTEGER PRIMARY KEY NOT NULL,
        event_no INTEGER NOT NULL,
        pulse_no INTEGER NOT NULL,
        dom_string INTEGER NOT NULL,
        dom_pmt INTEGER NOT NULL,
        dom_om INTEGER NOT NULL,
        dom_x REAL NOT NULL,
        dom_y REAL NOT NULL,
        dom_z REAL NOT NULL,
        dom_time INTEGER NOT NULL,
        dom_charge REAL NOT NULL,
        dom_lc INTEGER,
        dom_atwd INTEGER,
        dom_fadc INTEGER,
        dom_pulse_width INTEGER,
        SplitInIcePulses INTEGER,
        SRTInIcePulses INTEGER
    );
"""

sql_create_truth_table = """
    CREATE TABLE truth (
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
        pid INTEGER
    );
"""
sql_create_reconstruction_table = """
    CREATE TABLE reconstruction (
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
        pid INTEGER
    );
"""
sql_create_meta_table = """
    CREATE TABLE meta (
        event_no INTEGER PRIMARY KEY NOT NULL,
        file TEXT NOT NULL,
        idx INTEGER NOT NULL,
        level INTEGER,
        start_time TEXT,
        end_time TEXT
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
sql_update_reconstruction = """
    INSERT INTO reconstruction({}) VALUES ({})
""".format(
    ", ".join(list(reconstruction_columns.keys())),
    ", ".join(["?"] * len(list(reconstruction_columns.keys()))),
)
sql_update_meta = """
    INSERT INTO meta({}) VALUES ({})
""".format(
    ", ".join(list(meta_columns.keys())),
    ", ".join(["?"] * len(list(meta_columns.keys()))),
)


def create_sqlite_db(
    paths,
    level,
    no_files,
    particle_type,
    include_truth,
    include_reconstruction,
    write_to_db=True,
):
    dataset_root = paths["raw_files"]
    db = paths["fast_db"]
    gcd_file = [
        file for file in dataset_root.joinpath("gcd").iterdir() if file.is_file
    ][0]
    i3_files_root = dataset_root.joinpath("i3")

    i3_files = [file for file in i3_files_root.iterdir() if file.is_file()]
    if no_files == 1:
        i3_files = [i3_files[0]]
    elif no_files is None:
        i3_files = i3_files
    else:
        i3_files = i3_files[0:no_files]

    if write_to_db:
        create_db(db, particle_type, level, include_truth, include_reconstruction)

    last_event_no = [0]
    for i3_file in i3_files:
        now = datetime.now().strftime("%H:%M:%S")
        print("{}: Retrieving from {}".format(now, i3_file.name))
        last_event_no = [last_event_no[-1]]
        inputs = (i3_file, gcd_file, level, last_event_no, particle_type)
        features, truth, reconstruction, meta, last_event_no = i3_to_list_of_tuples(
            inputs
        )
        if write_to_db:
            with sqlite3.connect(str(db)) as con:
                cur = con.cursor()
                print(
                    "{}: inserting {} into DB".format(
                        datetime.now().strftime("%H:%M:%S"), i3_file.name
                    )
                )
                cur.executemany(sql_update_features, features)
                if include_truth:
                    try:
                        cur.executemany(sql_update_truth, truth)
                    except Exception as e:
                        print(str(e))
                        print(truth[0])
                if include_reconstruction:
                    cur.executemany(sql_update_reconstruction, reconstruction)
                cur.executemany(sql_update_meta, meta)
                con.commit()

    now = datetime.now().strftime("%H:%M:%S")
    print("{}: Done!".format(now))
