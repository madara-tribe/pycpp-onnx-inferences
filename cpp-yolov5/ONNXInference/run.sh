#!/bin/sh
cmake -B build
cmake --build build --config Release --parallel
cd build/src/
./yolov5s --model_path ../../models/yolov5s.onnx --image ../../images/bus.jpg --class_names ../../models/coco.names
