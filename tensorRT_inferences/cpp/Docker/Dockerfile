FROM nvcr.io/nvidia/pytorch:20.07-py3

# Reset default working directory
WORKDIR /workspace
# Install requried libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    wget sudo vim unzip zlib1g-dev \
    git pkg-config
 
RUN apt-get install -y p7zip-full



# Install Onnx  # protobuf
RUN apt-get install -y libprotobuf-dev protobuf-compiler python-setuptools
RUN cd /tmp && apt-get install -y libprotobuf-dev protobuf-compiler && \
    git clone --recursive https://github.com/onnx/onnx.git /tmp/onnx && \
    cd onnx && mkdir build && cd build && \
    cmake .. && \
    make -j $(nproc) && \
    make install 

RUN pip3 install --upgrade pip 
RUN pip3 install medpy 
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install torch torchvision torchtext torchaudio
RUN pip3 install --upgrade scikit-image
# Download onnx model
#RUN wget https://developer.download.nvidia.com/devblogs/speeding-up-unet.7z && \
    #apt-get install -y p7zip-full && 7z x speeding-up-unet.7z && \
    #cd unet && \
    #python3 create_network.py

RUN rm -rf /tmp/*

