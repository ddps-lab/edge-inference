FROM nvcr.io/nvidia/l4t-tensorflow:r32.6.1-tf2.5-py3 

RUN apt-get update && apt-get install -y git \
    vim \
    cmake \
    unzip \
    python3-pip \
    curl \
    git-lfs
 
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash

RUN python3 -m pip install -U pip \
    wheel \
    setuptools \
    setuptools_rust

RUN git clone https://github.com/ddps-lab/edge-inference.git

COPY ./requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
| tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get update

RUN git clone https://github.com/google-coral/pycoral.git
RUN git clone https://github.com/google-coral/test_data.git

RUN apt install -y python3-pycoral
