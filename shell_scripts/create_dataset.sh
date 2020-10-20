#!/bin/bash
docker run -v /home/ehrhorn/data:/home/icecube/data \
	-v /home/ehrhorn/work:/home/icecube/work \
	-v /home/ehrhorn/raw_files:/home/icecube/raw_files \
   ehrhorn/cubedev:0.2 \
	python -u /home/icecube/work/cubeflow/cubeflow/create_sql_db.py -n dev_numu_train_upgrade_step4_2020_00 -f 0 -c 1 -l 2
python -u ~/work/cubeflow/cubeflow/create_dataset.py -n dev_numu_train_upgrade_step4_2020_00 -r 0
