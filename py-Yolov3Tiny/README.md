## yolov3_to_onnx versions (Docker/requirements.txt)

```
tensorflow==1.13.2
keras==2.2.4
keras2onnx==1.5.1
onnxconverter-common==1.6.0
onnx==1.6.0
```


## Whole run of onnx tiny yolov3

```
./run.sh
```


# Convert keras yolov3-tiny to onnx

```
python3 keras_yolov3_onnx_convert.py ${YOLOV3-WEIGHT} 'anchors_classes/face_classes.txt' 'anchors_classes/tiny_yolo_anchors.txt' ${ONNX_NAME}
```

# Python onnx inference 

```
python3 py_onnx_inference.py ${ONNX_NAME}
```

# Python onnx inference result

## input

![person](https://user-images.githubusercontent.com/48679574/113898925-4c8d0480-979a-11eb-8858-77ae4af307e8.jpg)

## output

![output](https://user-images.githubusercontent.com/48679574/113898952-51ea4f00-979a-11eb-8cca-c88aff352ff4.jpg)


# References
[keras-yolov3](https://github.com/qqwweee/keras-yolo3)
