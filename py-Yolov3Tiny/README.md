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

![person](https://user-images.githubusercontent.com/48679574/163290206-d85a36d8-4b0f-4a8b-a3a6-395ae1c26f26.jpg)

## output

![output](https://user-images.githubusercontent.com/48679574/163290216-c0112aea-0c33-4bef-b12b-4fa6b23c2de9.jpg)


# References
[keras-yolov3](https://github.com/qqwweee/keras-yolo3)
