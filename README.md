# edge-inference
Evaluation of inference model performance on edge devices

### Nvidia L4T & TF2.5 & python3.6
        docker pull nvcr.io/nvidia/l4t-tensorflow:r32.6.1-tf2.5-py3

### Docker image build
    
        docker build -t edge-inference ./

### Docker container execution (Using GPU)
        docker run --privileged --gpus all -it edge-inference /bin/bash
