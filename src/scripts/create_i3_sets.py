import os
from pathlib import Path
import re
from shutil import copyfile
import stat

import numpy as np

from cubeflow.modules.helper_functions import open_pickle_file
from cubeflow.modules.helper_functions import get_project_root

raw_files_root = Path().home().joinpath("work").joinpath("raw_files")
i3_files_dict = raw_files_root.joinpath("files.pkl")
i3_files = open_pickle_file(str(i3_files_dict))
project_root = get_project_root()
shell_script_name = "get_files_from_hep.sh"
shell_script_path = project_root.joinpath("shell_scripts").joinpath(shell_script_name)
insert_i3_in_filename_name = "insert_i3_in_filename.sh"
insert_i3_in_filename_path = project_root.joinpath("shell_scripts").joinpath(
    insert_i3_in_filename_name
)

sets = {
    "dev_muongun_train_l2_2020": {
        "sizes": ["01", "02", "03"],
        "fileset": "muongun_lvl2",
        "pid": "13",
    },
    "dev_numu_train_l2_2020": {
        "sizes": ["01", "02", "03"],
        "fileset": "genie_lvl2",
        "pid": "14",
    },
    "dev_nue_train_l2_2020": {
        "sizes": ["01", "02", "03"],
        "fileset": "genie_lvl2",
        "pid": "12",
    },
    "dev_numu_train_l5_2020": {
        "sizes": ["01", "02", "03"],
        "fileset": "genie_lvl5",
        "pid": "14",
    },
    "dev_numu_predict_l2_2020": {
        "sizes": ["01"],
        "fileset": "genie_lvl2",
        "pid": "14",
    },
    "dev_data_predict_l2_2020": {
        "sizes": ["01"],
        "fileset": "data_lvl2",
        "pid": "data",
    },
    "dev_numu_train_upgrade_step4_2020": {
        "sizes": ["00", "01", "02"],
        "fileset": "upgrade_step4",
        "pid": "14",
    },
    "dev_numu_predict_upgrade_step4_2020": {
        "sizes": ["01", "02"],
        "fileset": "upgrade_step4",
        "pid": "14",
    },
}


def take_random_files(files, size):
    rng = np.random.default_rng(seed=41)
    choices = rng.choice(files, size=size, replace=False)
    return choices


filesets = {}
for key, values in sets.items():
    for size in values["sizes"]:
        filesets[key + "_" + size] = {}
        if size == "00":
            files = i3_files[values["fileset"]][values["pid"]]["files"]
            no_files = 300
        if size == "01":
            files = i3_files[values["fileset"]][values["pid"]]["files"]
            no_files = 60
        elif size == "02":
            files = filesets[key + "_01"]["files"]
            no_files = 30
        elif size == "03":
            files = filesets[key + "_02"]["files"]
            no_files = 15
        if "predict" in key and values["pid"] != "data":
            nixed_files = filesets[key.replace("predict", "train") + "_01"]["files"]
            files = i3_files[values["fileset"]][values["pid"]]["files"]
            files = [file for file in files if file not in nixed_files]
        choices = take_random_files(files, no_files)
        filesets[key + "_" + size]["files"] = choices
        filesets[key + "_" + size]["gcd"] = i3_files[values["fileset"]][values["pid"]][
            "gcd"
        ]

train_files = set(filesets["dev_numu_train_l2_2020_01"]["files"])
prediction_files = set(filesets["dev_numu_predict_l2_2020_01"]["files"])
intersection = list(train_files & prediction_files)
print("intersection:", len(intersection))


for fileset, values in filesets.items():
    fileset_dir = raw_files_root.joinpath(fileset)
    fileset_dir.mkdir(exist_ok=True)
    gcd_dir = fileset_dir.joinpath("gcd")
    gcd_dir.mkdir(exist_ok=True)
    i3_files_dir = fileset_dir.joinpath("i3")
    i3_files_dir.mkdir(exist_ok=True)
    copied_shell_script_path = fileset_dir.joinpath(shell_script_name)
    copyfile(str(shell_script_path), str(copied_shell_script_path))
    st = os.stat(str(copied_shell_script_path))
    os.chmod(str(copied_shell_script_path), st.st_mode | stat.S_IEXEC)
    copied_insert_i3_in_filename_path = fileset_dir.joinpath(insert_i3_in_filename_name)
    copyfile(str(insert_i3_in_filename_path), str(copied_insert_i3_in_filename_path))
    st = os.stat(str(copied_insert_i3_in_filename_path))
    os.chmod(str(copied_insert_i3_in_filename_path), st.st_mode | stat.S_IEXEC)
    with open(str(fileset_dir.joinpath("files.txt")), "w") as text_file:
        for row in values["files"]:
            text_file.write(row + "\n")
    with open(str(fileset_dir.joinpath("gcd.txt")), "w") as text_file:
        text_file.write(values["gcd"])

key = "dev_numu_predict_l5_2020"
size = "01"
filesets[key + "_" + size] = {}
files = filesets["dev_numu_predict_l2_2020_01"]["files"]
files_root = i3_files["genie_lvl5"]["14"]["files"][0].split(
    "oscNext_genie_level5_v01.01_pass2.140000"
)[0]
file_nos = []
regex_pattern = r"(?<=\.|_)[0]{2}(.*)[0-9]{2}"
for file in files:
    file_nos.append(re.search(regex_pattern, file).group(0))
new_file_name = (
    "oscNext_genie_level5_v01.01_pass2.140000.001549__retro_crs_prefit.i3.zst"
)
files = []
for file_no in file_nos:
    temp = re.sub(regex_pattern, str(file_no), new_file_name)
    files.append(files_root + temp)

filesets[key + "_" + size]["files"] = np.array(files)
filesets[key + "_" + size]["gcd"] = i3_files["genie_lvl5"]["14"]["gcd"]

fileset = key + "_" + size
fileset_dir = raw_files_root.joinpath(fileset)
fileset_dir.mkdir(exist_ok=True)
gcd_dir = fileset_dir.joinpath("gcd")
gcd_dir.mkdir(exist_ok=True)
i3_files_dir = fileset_dir.joinpath("i3")
i3_files_dir.mkdir(exist_ok=True)
copied_shell_script_path = fileset_dir.joinpath(shell_script_name)
copyfile(str(shell_script_path), str(copied_shell_script_path))
st = os.stat(str(copied_shell_script_path))
os.chmod(str(copied_shell_script_path), st.st_mode | stat.S_IEXEC)
copied_insert_i3_in_filename_path = fileset_dir.joinpath(insert_i3_in_filename_name)
copyfile(str(insert_i3_in_filename_path), str(copied_insert_i3_in_filename_path))
st = os.stat(str(copied_insert_i3_in_filename_path))
os.chmod(str(copied_insert_i3_in_filename_path), st.st_mode | stat.S_IEXEC)
with open(str(fileset_dir.joinpath("files.txt")), "w") as text_file:
    for row in filesets[fileset]["files"]:
        text_file.write(row + "\n")
with open(str(fileset_dir.joinpath("gcd.txt")), "w") as text_file:
    text_file.write(filesets[fileset]["gcd"])
