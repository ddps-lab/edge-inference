# docker pull nvcr.io/nvidia/l4t-tensorflow:r32.6.1-tf2.5-py3

FROM nvcr.io/nvidia/l4t-tensorflow:r32.6.1-tf2.5-py3 

RUN apt-get update && apt-get install -y git \
    vim \
    cmake \
    unzip \
    curl \
    git-lfs
 
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash

RUN python3 -m pip install -U pip \
    wheel \
    setuptools \
    setuptools_rust

RUN git clone https://github.com/ddps-lab/edge-inference.git

RUN pip3 install -r ./edge-inference/requirements.txt
