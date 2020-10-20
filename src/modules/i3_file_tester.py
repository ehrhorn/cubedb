from datetime import datetime
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


def fetch_events(frame):
    if frame["I3EventHeader"].event_id == 2638:
        print("hej")


def i3_to_list_of_tuples(gcd_file, i3_file):
    tray = I3Tray()
    tray.AddModule("I3Reader", "reader", FilenameList=[str(gcd_file)] + [str(i3_file)])
    tray.Add(
        fetch_events, "fetch_events",
    )
    tray.Execute()
    tray.Finish()


gcd_file = "/groups/icecube/stuttard/data/oscNext/pass2/gcd/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz"
i3_file = "/groups/icecube/stuttard/data/oscNext/pass2/genie/level5_v01.01/140000/oscNext_genie_level5_v01.01_pass2.140000.000713__retro_crs_prefit.i3.zst"

i3_to_list_of_tuples(gcd_file, i3_file)
