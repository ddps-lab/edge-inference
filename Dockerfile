FROM nvcr.io/nvidia/l4t-tensorflow:r32.6.1-tf2.5-py3 

## CNN, NLP library install 
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

# USB Coral TPU library install
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get update

RUN apt-get install -y gasket-dkms \
    libedgetpu1-std \
    usbutils
RUN apt-get install -y python3-edgetpu

<<<<<<< HEAD
RUN pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_aarch64.whl

# model, dataset, inference code git repo clone
RUN git clone https://github.com/ddps-lab/edge-inference.git

=======
>>>>>>> 45a209bda01eb1ae4b5c7ceddfee1422c5ae4fba
COPY ./requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
