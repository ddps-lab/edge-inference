# edge-inference
Evaluation of inference model performance on edge devices

### Nvidia L4T & TF2.5 & python3.6
        docker pull nvcr.io/nvidia/l4t-tensorflow:r32.6.1-tf2.5-py3

### File Download
        curl -O https://raw.githubusercontent.com/ddps-lab/edge-inference/main/Dockerfile
        curl -O https://raw.githubusercontent.com/ddps-lab/edge-inference/main/requirements.txt

### Docker image build
        docker build -t edge-inference ./

### Docker container execution (Using GPU)
        docker run --privileged --gpus all --shm-size 10G -it edge-inference /bin/bash

### Docker container execution (Using TPU)
	docker run --privileged -it edge-inference /bin/bash
