import argparse

import modules.helper_functions as helper_functions

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--dataset_name", help="dataset name")
parser.add_argument("-r", "--reconstruction", help="reconstruction")
args = parser.parse_args()

dataset_name = args.dataset_name
paths = helper_functions.get_dataset_paths(dataset_name)

reconstruction = int(args.reconstruction)

helper_functions.print_message_with_time("Creating dataset")

helper_functions.index_db(dataset_name)
if "predict" in dataset_name:
    helper_functions.print_message_with_time("Moving truth to prediction...")
    helper_functions.truth_to_prediction_db(dataset_name)
helper_functions.create_sets(dataset_name)
helper_functions.create_transformers(dataset_name)
helper_functions.create_dataset_distribution_histograms(dataset_name)
helper_functions.transform_db(dataset_name)
helper_functions.create_dataset_distribution_histograms(dataset_name, transform=True)
helper_functions.move_file(paths["fast_db"], paths["db"])

if "predict" in dataset_name and reconstruction == 1:
    helper_functions.print_message_with_time("Calculating errors...")
    helper_functions.calculate_errors(
        "reconstruction", dataset_name, ["energy", "time", "direction", "vertex"]
    )

helper_functions.print_message_with_time("Created dataset")
