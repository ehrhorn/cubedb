#!/bin/bash
python -u ./src/create_sql_db.py -n $1 -q candidate_events.sql
# if ! command -v docker &> /dev/null
# then
#     echo "Docker could not be found"
#     singularity run --no-home -B $HOME/data:/home/icecube/data,$HOME/work:/home/icecube/work \
#         docker://ehrhorn/cubedev:0.2 \
#         python -u /home/icecube/work/cubedb/src/create_sql_db.py -n $1 -q /home/icecube/work/cubedb/candidate_events.sql
#     python -u $HOME/work/cubedb/src/create_dataset.py -n $1
# else
#     docker run -v $HOME/data:/home/icecube/data \
#         -v $HOME/work:/home/icecube/work \
#         ehrhorn/cubedev:0.2 \
#         python -u /home/icecube/work/cubedb/src/create_sql_db.py -n $1 -q /home/icecube/work/cubedb/candidate_events.sql
#     python -u $HOME/work/cubedb/src/create_dataset.py -n $1
# fi