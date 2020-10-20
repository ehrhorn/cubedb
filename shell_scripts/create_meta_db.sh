#!/bin/bash
docker run -v /home/ehrhorn/data:/home/icecube/data \
	-v /home/ehrhorn/work:/home/icecube/work \
	-v /home/ehrhorn/raw_files:/home/icecube/raw_files \
   ehrhorn/cubedev:0.2 \
	python -u /home/icecube/work/cubeflow/cubeflow/create_i3_sets_2.py