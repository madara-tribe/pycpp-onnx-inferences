# !/bin/sh
# convert 
cd convert
python3 tfkeras_model_save.py
python3 tfkeras2tensorRT.py FP32
python3 tfkeras2tensorRT.py FP16
python3 tfkeras2tensorRT.py INT8
cd ../

# tfkeras inference
python3 tfkeras_inference.py

# trt inference
python3 trt_inference.py FP32
python3 trt_inference.py FP16
python3 trt_inference.py INT8

# output each model info
./model_info.sh

# reset
rm -rf */__pycache__ __pycache__
rm -rf model_weights/TRTFP32 model_weights/TRTFP16 model_weights/TRTINT8 model_weights/tfkeras_model
