FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN apt-get update --fix-missing && apt-get install -y wget \
    libsndfile1 sox git git-lfs bash build-essential curl ca-certificates python3 python3-pip

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir mkl torch  

RUN python3 -m pip --no-cache-dir install fairseq@git+https://github.com//pytorch/fairseq.git@1b61bbad327d2bf32502b3b9a770b57714cc43dc#egg=fairseq

RUN python3 -m pip --no-cache-dir install git+https://github.com/s3prl/s3prl.git@185e4b060cd96ce5911e258c2fde74a2e8246308#egg=s3prl

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1