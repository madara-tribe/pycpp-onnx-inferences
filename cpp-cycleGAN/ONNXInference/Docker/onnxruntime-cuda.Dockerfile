FROM nvcr.io/nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
# FROM nvcr.io/nvidia/tensorrt:20.09-py3

ENV OPENCV_VERSION=4.5.3
ARG ONNXRUNTIME_VERSION=1.6.0
ARG NUM_JOBS=8
ENV DEBIAN_FRONTEND noninteractive

# Install package dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential software-properties-common \
        autoconf automake libtool pkg-config ca-certificates wget \
        git curl libjpeg-dev libpng-dev language-pack-en \
        locales locales-all python3 \
        python3-py python3-dev python3-pip python3-numpy \
        python3-pytest python3-setuptools libprotobuf-dev \
        protobuf-compiler zlib1g-dev swig vim gdb valgrind \
        libsm6 libxext6 libxrender-dev unzip sudo


RUN cd /usr/local/bin && \
    ln -s /usr/bin/python3 python && \
    ln -s /usr/bin/pip3 pip && \
    pip install --upgrade pip setuptools wheel

# install CMake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.16.8/cmake-3.16.8-Linux-x86_64.sh && \
    chmod +x cmake-3.16.8-Linux-x86_64.sh && \
    ./cmake-3.16.8-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license
RUN rm -rf /tmp/*

# install Opencv
RUN cd /tmp \
    && wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
	&& unzip ${OPENCV_VERSION}.zip \
	&& rm ${OPENCV_VERSION}.zip
RUN cd /tmp/opencv-${OPENCV_VERSION} \
	&& mkdir build \
	&& cd build \
	&& cmake \
		-DCMAKE_BUILD_TYPE=RELEASE \
		-DCMAKE_INSTALL_PREFIX=/usr/local \
		-DENABLE_PRECOMPILED_HEADERS=OFF ..
RUN cd /tmp/opencv-${OPENCV_VERSION}/build \
    && make -j $(nproc) \
	&& make install \
	&& ldconfig

# Install ONNX Runtime
RUN pip install pytest==6.2.1 onnx==1.8.0
RUN cd /tmp && \
    git clone --recursive --branch v${ONNXRUNTIME_VERSION} https://github.com/Microsoft/onnxruntime && \
    cd onnxruntime && \
    ./build.sh \
        --cuda_home /usr/local/cuda \
        --cudnn_home /usr/lib/x86_64-linux-gnu/ \
        --use_cuda \
        # --use_tensorrt \
        # --tensorrt_home /usr/lib/x86_64-linux-gnu/ \
        --config RelWithDebInfo \
        --build_shared_lib \
        --build_wheel \
        --skip_tests \
        --parallel ${NUM_JOBS} && \
    cd build/Linux/RelWithDebInfo && \
    make install && \
    pip install dist/*
RUN rm -rf /tmp/*
