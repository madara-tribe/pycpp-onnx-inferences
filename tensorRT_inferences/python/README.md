# python tensorRT inference

python tensorRT inference converted from age-gender-model
- [age-gender-model](https://github.com/madara-tribe/yolo_age_gender_model)

# versions && env info
- cuda 10.1
- ubuntu 18.04
- python 3.6
- tensorflow==2.0.0

```
$ nvidia-smi
 NVIDIA-SMI 430.64       Driver Version: 430.64       CUDA Version: 10.1
 
$ ./info/cuda_info.sh 
>>>
lrwxrwxrwx  1 root root    9 Jul 29  2019 cuda -> cuda-10.1
drwxr-xr-x 22 root root 4096 May  5 22:12 cuda-10.1
drwxr-xr-x  3 root root 4096 May  5 22:12 cuda-10.2
ii  graphsurgeon-tf                                             5.1.5-1+cuda10.1                    amd64        GraphSurgeon for TensorRT package
ii  libnvinfer5                                                 5.1.5-1+cuda10.1                    amd64        TensorRT runtime libraries
ii  uff-converter-tf                                            5.1.5-1+cuda10.1                    amd64        UFF converter for TensorRT package
```

# convert tf-keras to tensorRT (FP32, FP16, INT8)

At first, save tf-keras model
```$ python3 convert/tfkeras_model_save.py```

Next, convert tf-keras to tensorRT FP32, FP16, INT8
```zsh
cd convert
python3 tfkeras2tensorRT.py FP32
python3 tfkeras2tensorRT.py FP16
python3 tfkeras2tensorRT.py INT8
```

## targrt image

![9_1_2_face](https://user-images.githubusercontent.com/48679574/117323063-87817580-aec9-11eb-8b22-727c217f8d7d.jpg)


# inference performance (on CPU)

whole inference run
```
./whole_inference_run.sh
```

## tf-keras inference 
```zsh
# batched_input shape:  (1, 299, 299, 3)
tfkeras gender M and age 33
Inference Latency (milliseconds) is 3066.9949054718018 [ms]
```

## tenserRT FP32 inference
```
TRTFP32 gender M and age 33
Inference Latency (milliseconds) is 47.1796989440918 [ms]
```

## tenserRT FP16 inference
```zsh
TRTFP16 gender M and age 33
Inference Latency (milliseconds) is 45.10211944580078 [ms]
```

## tenserRT INT8 inference
```zsh
RTINT8 gender M and age 33
Inference Latency (milliseconds) is 45.09305953979492 [ms]
```

# References

- [tensorflow/tensorrt](https://github.com/tensorflow/tensorrt)
- [NVIDIA TENSERRT DOCUMMENT](https://docs.nvidia.com/deeplearning/tensorrt/archives/index.html)

# Note (2021/05)

[Starting from Tensorflow 1.9.0, it already has TensorRT inside the tensorflow contrib, but some issues are encountered.](https://medium.com/@ardianumam/installing-tensorrt-in-ubuntu-dekstop-1c7307e1dcf6)
