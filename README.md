# onnx models inference by python and c++


# How fast to predict 

## c++ yolov5
```zsh
# yolov5s.onnx
process time: 345[ms]

# yolov5m.onnx
process time: 636[ms]
```

## python "age-gender-model"

```zsh
onnx input shape (1, 299, 299, 3)
start calculation
pred_age is  27 & gender is Female
Inference Latency (milliseconds) is 29.436588287353516 [ms]
```

## c++ CycleGAN
```zsh
# c++
ONNX Inference Latency: 1.84 ms

# python
Inference Latency (milliseconds) is 45.3539218902588 [ms]
```
