FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /root
ENV PYTHON_VERSION 3.6.0

RUN apt-get update && \
	apt-get install -y sudo \
	build-essential curl libcurl4-openssl-dev libssl-dev wget git tzdata \
        tk-dev vim unzip

RUN apt-get install tree


RUN apt-get update && apt-get upgrade -y && \
    apt install -y  --no-install-recommends make cmake gcc git g++ unzip wget build-essential zlib1g-dev libffi-dev libssl-dev && \
    apt clean && \
    wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar zxf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure && \
    make altinstall

RUN set -x && \
    wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh && \
    bash Anaconda3-2019.10-Linux-x86_64.sh -b && \
    rm Anaconda3-2019.10-Linux-x86_64.sh

ENV PATH $PATH:/root/anaconda3/bin

# install pip
RUN /root/anaconda3/bin/pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN /root/anaconda3/bin/pip install -r requirements.txt
#CMD [ "/bin/bash" ]
