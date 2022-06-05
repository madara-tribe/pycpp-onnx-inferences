# cpp_onnx_tensorRT

# Version
- Python ==3.6.10 Anaconda, Inc. (path: ```/opt/conda/bin/python3```)
- Pytorch == 1.9.0+cu102
- cuda == cuda_11 (``` cuda_11.0_bu.TC445_37.28540450_0```)
- TensorRT == 7.1.3-1+cuda11.0 (```$ dpkg -l | grep nvinfer```)
- onnxruntime == 1.3.0
- onnx==1.6.0


Your Nvidia Driver failed, if ```nvidia-smi``` command can not be used. you must use ```nvidia-smi``` command.

# Docker
Use Nvidia Docker or customed Dockerfile
```
docker pull nvcr.io/nvidia/pytorch:20.07-py3
docker run --gpus all -it nvcr.io/nvidia/pytorch:20.07-py3

or 

cd Docker
make run && make in
```

# Inference

## dataset 
```
# download onnx file
cd unet
./download.sh

# create pb file
mkdir test_dataset
python3 unet/prepareData.py --input_image test_images/brain_mri_4947.tif --input_tensor test_dataset/input_0.pb --output_tensor test_dataset/output_0.pb
```


## trt compile
source code is download as ```git clone https://github.com/parallel-forall/code-samples.git```
```
# git clone https://github.com/parallel-forall/code-samples.git
cd code-samples/posts/TensorRT-introduction
# Compile the TensorRT C++ code
make clean && make
```
## cpp trt onnx inference
```
cd code-samples/posts/TensorRT-introduction
./simpleOnnx_1 ../../../onnx_unet/unet.onnx ../../../test_data_set_0/input_0.pb
make clean
```

# References
- [TensorRT 7 で 推論の高速化](https://qiita.com/k_ikasumipowder/items/cec4d11261139d79070b)
