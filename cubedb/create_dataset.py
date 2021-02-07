import argparse
from datetime import datetime

import modules.helper_functions as helper_functions
from modules.save_random_events import random_events_saver

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--dataset_name", help="dataset name")
args = parser.parse_args()

dataset_name = args.dataset_name
paths = helper_functions.get_dataset_paths(dataset_name)

helper_functions.print_message_with_time("Creating dataset")

helper_functions.index_db(dataset_name)
try:
    helper_functions.print_message_with_time("Moving truth to prediction...")
    helper_functions.truth_to_prediction_db(dataset_name)
except Exception as e:
    print("{}: Encountered error: {}".format(datetime.now(), str(e)))
helper_functions.create_sets(dataset_name)
helper_functions.create_transformers(dataset_name)
helper_functions.create_dataset_distribution_histograms(dataset_name)
helper_functions.transform_db(dataset_name)
helper_functions.create_dataset_distribution_histograms(dataset_name, transform=True)
helper_functions.move_file(paths["fast_db"], paths["db"])
random_events_saver(dataset_name, limit=1e4)

helper_functions.print_message_with_time("Created dataset")
