#!/bin/sh
# conda create -n <name> python==3.8
pip3 install -r requirements.txt
python3 convert.py --weights weights/yolov5s.pt --include onnx #torchscript

