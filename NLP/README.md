- nvidia docker 이미지 다운로드
    
    ```bash
    docker pull nvcr.io/nvidia/l4t-tensorflow:r32.6.1-tf2.5-py3
    ```
    
- docker 컨테이너 실행
    
    ```bash
    docker run --name unho --gpus all -it nvcr.io/nvidia/l4t-tensorflow:r32.6.1-tf2.5-py3 /bin/bash
    ```
    
- 필요 패키지 및 파일 설치
    
    ```bash
    apt-get update
    apt-get install git -y
    apt-get install curl -y
    apt-get install git-lfs
    
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
    
    python3 -m pip install -U pip wheel setuptools setuptools_rust
    pip3 install transformers datasets
    pip3 install bert-for-tf2
    pip3 install tensorflow_hub
    
    git clone https://github.com/ddps-lab/edge-inference.git
    ```
