#!/bin/bash

is_dockerfile=$(ls Dockerfile)

if [ -z $is_dockerfile ] 
then
        echo "Dockerfile not exist!"
        exit
fi

image_name="edge-tf-serving"
image_id=$(docker images -aq $image_name)

if [ -z "$image_id" ] 
then
        docker build -t edge-tf-serving .
fi

docker run --rm \
        --device /dev/nvhost-ctrl \
        --device /dev/nvhost-ctrl-gpu \
        --device /dev/nvhost-prof-gpu \
        --device /dev/nvmap \
        --device /dev/nvhost-gpu \
        --device /dev/nvhost-as-gpu \
        -p 8500:8500 \
        -v ~/edge-inference/CNN/model/:/models/model/ \
        edge-tf-serving:latest