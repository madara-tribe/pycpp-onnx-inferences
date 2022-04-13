import os
os.environ['TF_KERAS'] = '1'
import numpy as np
import keras2onnx
from keras2onnx import convert_keras
import onnxruntime
import onnx
from models.age_gender_model import load_model

weight_dir = 'weights'
weight_name = 'ep10age_gender_299x299.hdf5'
onnx_output_path = 'models'
OUTPUT_ONNX_MODEL_NAME = 'age_gender_model.onnx'
OPSET = 11

def main():
    onnx_model_file_name = OUTPUT_ONNX_MODEL_NAME
    target_opset = OPSET
    models = load_model()
    models.load_weights(os.path.join(weight_dir, weight_name))
    print(models.name)
    
    onnx_model = convert_keras(models, models.name, target_opset=target_opset, channel_first_inputs=models.inputs)
    onnx.save(onnx_model, os.path.join(onnx_output_path, onnx_model_file_name))
    print("success to output "+onnx_model_file_name)

if __name__=='__main__':
    main()
