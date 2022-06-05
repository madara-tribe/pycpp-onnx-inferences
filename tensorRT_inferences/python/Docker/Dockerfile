FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION 3.7.0
ARG OS=ubuntu1804
ARG TAG=cuda10.1-trt5.1.5.0-ga-20190427

WORKDIR /home/app
RUN apt-get update && apt-get upgrade -y
RUN apt install -y \
    sudo tzdata vim git cmake wget unzip build-essential libbz2-dev tk-dev


COPY nv-tensorrt-repo-${OS}-${TAG}_1-1_amd64.deb /tmp
RUN dpkg -i /tmp/nv-tensorrt-repo-${OS}-${TAG}_1-1_amd64.deb && \
    apt-key add /var/nv-tensorrt-repo-${TAG}/7fa2af80.pub

RUN apt-get update && apt-get install libnvinfer5

RUN apt-get install -y python3 python3-pip
RUN apt-get install -y python-opencv python3-opencv

RUN pip3 install --upgrade pip 
RUN pip3 install pillow matplotlib scikit-image opencv-python
RUN pip3 install tensorflow==2.0.0 h5py==2.10.0
RUN apt-get install -y uff-converter-tf
CMD [ "/bin/bash" ]
