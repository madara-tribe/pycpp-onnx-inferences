#!/bin/sh
ONNX_NAME=yolov3_onnx.opt.onnx
TINY_YOLO_WEIGHT=weights/tiny_yolov3_face.h5
python3 keras_yolov3_onnx_convert.py ${TINY_YOLO_WEIGHT} anchors_classes/face_classes.txt anchors_classes/tiny_yolo_anchors.txt ${ONNX_NAME}

# inference
python3 python_inference.py ${ONNX_NAME}

rm -rf __pycache__ */__pycache__ # final_model.h5 output.jpg yolov3-face.opt.onnx
