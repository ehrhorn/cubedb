#!/bin/bash
if ! command -v docker &> /dev/null
then
    echo "Docker could not be found"
    singularity run --no-home -H /home/icecube -B $HOME/data:/home/icecube/data,$HOME/work:/home/icecube/work,/groups/icecube/stuttard/data:/groups/icecube/stuttard/data \
        docker://ehrhorn/cubedb:0.1 \
        python -u /home/icecube/work/cubedb/src/create_sql_db.py -n $1 -q /home/icecube/work/cubedb/candidate_events.sql
    singularity run --no-home -H /home/icecube -B $HOME/data:/home/icecube/data,$HOME/work:/home/icecube/work,/groups/icecube/stuttard/data:/groups/icecube/stuttard/data \
        docker://ehrhorn/cubedb:0.1 \
        python -u /home/icecube/work/cubedb/src/create_dataset.py -n $1
else
    docker run -v $HOME/data:/home/icecube/data \
        -v $HOME/work:/home/icecube/work \
        ehrhorn/cubedb:0.2 \
        python -u /home/icecube/work/cubedb/src/create_sql_db.py -n $1 -q /home/icecube/work/cubedb/candidate_events.sql
    docker run -v $HOME/data:/home/icecube/data \
        -v $HOME/work:/home/icecube/work \
        ehrhorn/cubedb:0.2 \
        python -u /home/icecube/work/cubedb/src/create_dataset.py -n $1
fi