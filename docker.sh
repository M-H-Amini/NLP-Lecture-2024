#!/bin/bash

docker build -t mh_pytorch .

# Run the Docker container with GPU access
docker run --gpus device=2 -v ${PWD}:/usr/src/app -it --name mh_pytorch_container mh_pytorch bash -l