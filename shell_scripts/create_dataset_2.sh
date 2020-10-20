#!/bin/bash
python -u ./cubeflow/create_sql_db_2.py -n dev_numu_train_l5_retro_001 -q candidate_events.sql
# if ! command -v docker &> /dev/null
# then
#     echo "Docker could not be found"
#     singularity run -B $HOME/data:/home/icecube/data,$HOME/work:/home/icecube/work \
#         docker://ehrhorn/cubedev:0.2 \
#         python -u /home/icecube/work/cubeflow/cubeflow/create_sql_db_2.py -n singularity_test -q /home/icecube/work/cubeflow/candidate_events.sql
#     python -u $HOME/work/cubeflow/cubeflow/create_dataset_2.py -n singularity_test
# else
#     docker run -v $HOME/data:/home/icecube/data \
#         -v $HOME/work:/home/icecube/work \
#         ehrhorn/cubedev:0.2 \
#         python -u /home/icecube/work/cubeflow/cubeflow/create_sql_db_2.py -n singularity_test -q /home/icecube/work/cubeflow/candidate_events.sql
#     python -u $HOME/work/cubeflow/cubeflow/create_dataset_2.py -n singularity_test
# fi