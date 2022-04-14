# py_onnx_inference

# Version
```zsh
python==3.7.0
tensorflow 2.3.0
keras 2.3.1
keras2onnx 1.8.0
onnxconverter-common==1.6.0
onnx==1.6.0
```


# Convert age-gender-model to onnx


## tf.keras age-gender-model memory
```zsh
Total params: 21,808,931
Trainable params: 21,774,499
Non-trainable params: 34,432
```


## target image

![face](https://user-images.githubusercontent.com/48679574/163290201-bccf611e-130a-420c-a7b4-51401cd7652a.jpg)



# ONNX Python Inference Latency (millisecond[m/s])

```txt
onnx input shape (1, 299, 299, 3)
start calculation
pred_age is  27 & gender is Female
Inference Latency (milliseconds) is 29.436588287353516 [ms]
```
