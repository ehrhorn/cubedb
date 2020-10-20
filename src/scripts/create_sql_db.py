import argparse
from datetime import datetime

from modules.sqlite_creator import create_sqlite_db
from modules.helper_functions import get_dataset_paths
from modules.get_event_predictions import match_event_predictions

# from modules.sqlite_creator import get_event_predictions

parser = argparse.ArgumentParser()
parser.add_argument("-n")
parser.add_argument("-f", nargs="?", const=None)
parser.add_argument("-c", nargs="?", const=None)
parser.add_argument("-l", nargs="?", const=None)
args = parser.parse_args()

dataset_name = args.n

if "_l2_" in dataset_name:
    level = 2
    include_truth = True
    include_reconstruction = False
elif "_l4_" in dataset_name:
    level = 4
    include_truth = True
    include_reconstruction = False
elif "_l5_" in dataset_name:
    level = 5
    include_truth = True
    include_reconstruction = True
elif "_data_" in dataset_name:
    level = None
    include_truth = False
    include_reconstruction = False
elif "_step4_" in dataset_name:
    level = None
    include_truth = True
    include_reconstruction = False

if "_numu_" in dataset_name or "_nue_" in dataset_name or "_nutau_" in dataset_name:
    particle_type = "neutrino"
elif "_muongun_" in dataset_name:
    particle_type = "muon"
elif "_data_" in dataset_name:
    particle_type = None

if args.f == "0":
    no_files = None
else:
    no_files = int(args.f)

if args.c is None:
    create_db = False
else:
    create_db = bool(int(args.c))

if args.l is None:
    match_level = None
else:
    match_level = int(args.l)

paths = get_dataset_paths(dataset_name)

now = datetime.now().strftime("%H:%M:%S")
print(
    "{}: Beginning creating SQL db for dataset {}; particle type: {}, level {}, {} files".format(
        now, dataset_name, particle_type, level, no_files
    )
)

create_sqlite_db(
    paths,
    level,
    no_files,
    particle_type,
    include_truth,
    include_reconstruction,
    write_to_db=create_db,
)


if "predict" in dataset_name and match_level is not None:
    match_event_predictions(dataset_name, match_level, particle_type)

now = datetime.now().strftime("%H:%M:%S")
print("{}: Ended creating SQL db for dataset {}".format(now, dataset_name))
