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

# WORKDIR /edge-inference/NLP/model/
# RUN curl -O https://edge-inference.s3.us-west-2.amazonaws.com/bert_imdb_model.h5
# RUN curl -O https://edge-inference.s3.us-west-2.amazonaws.com/rnn_imdb_model.h5
# RUN curl -O https://edge-inference.s3.us-west-2.amazonaws.com/lstm_imdb_model.h5
# RUN curl -O https://edge-inference.s3.us-west-2.amazonaws.com/distilbert_sst2_model.h5
# WORKDIR /edge-inference/NLP/dataset/
# RUN curl -O https://edge-inference.s3.us-west-2.amazonaws.com/bert_dataset.zip
# RUN unzip bert_dataset.zip && rm -rf bert_dataset.zip
# WORKDIR /

COPY ./requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
