#!/bin/bash

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