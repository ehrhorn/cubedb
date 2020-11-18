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


def fetch_geom(frame, inputs):
    gcd_dict = inputs
    if len(gcd_dict[list(gcd_dict.keys())[0]]) > 0:
        return False
    dom_geom = frame["I3Geometry"].omgeo
    for entry in dom_geom:
        om_key = entry[0]
        om_geom = entry[1]
        om_position = om_geom.position
        om_orientation = om_geom.orientation
        om_area = om_geom.area
        om_type = om_geom.omtype
        gcd_dict["string"].append(om_key[0])
        gcd_dict["dom"].append(om_key[1])
        gcd_dict["pmt"].append(om_key[2])
        gcd_dict["dom_x"].append(om_position.x)
        gcd_dict["dom_y"].append(om_position.y)
        gcd_dict["dom_z"].append(om_position.z)
        gcd_dict["pmt_x"].append(om_orientation.x)
        gcd_dict["pmt_y"].append(om_orientation.y)
        gcd_dict["pmt_z"].append(om_orientation.z)
        gcd_dict["pmt_area"].append(om_area)
        gcd_dict["pmt_type"].append(om_type)


def gcd_to_dict(gcd_file, i3_file, gcd_dict):
    tray = I3Tray()
    tray.AddModule("I3Reader", "reader", FilenameList=[str(gcd_file)] + [str(i3_file)])
    tray.Add(
        fetch_geom,
        "fetch_geom",
        inputs=(gcd_dict),
    )
    tray.Execute()
    tray.Finish()
    return gcd_dict


def create_gcd_db(query, paths):
    meta_db = Path().home().joinpath("work").joinpath("datasets").joinpath("meta.db")
    with sqlite3.connect(str(meta_db)) as con:
        events = pd.read_sql(query, con)
    gcd_files = events["gcd_files"].unique()
    i3_temp_files = []
    for gcd_file in gcd_files:
        temp_df = events[events["gcd_files"] == gcd_file].copy()
        i3_temp_file = temp_df["files"].iloc[0].split(",")[0]
        i3_temp_files.append(i3_temp_file)
    for i, gcd_file in enumerate(gcd_files):
        gcd_dict = {
            "string": [],
            "dom": [],
            "pmt": [],
            "dom_x": [],
            "dom_y": [],
            "dom_z": [],
            "pmt_x": [],
            "pmt_y": [],
            "pmt_z": [],
            "pmt_area": [],
            "pmt_type": [],
        }
        out_file = paths["meta"] / (gcd_file.split("/")[-1] + ".db")
        gcd_dict = gcd_to_dict(gcd_file, i3_temp_files[i], gcd_dict)
        dataframe = pd.DataFrame().from_dict(data=gcd_dict)
        with sqlite3.connect(str(out_file)) as con:
            dataframe.to_sql(
                name="geometry",
                con=con,
                if_exists="replace",
                index=True,
                index_label="row",
            )
