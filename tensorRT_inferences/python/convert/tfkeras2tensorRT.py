import os
import time
import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from trt_inference import call_batched_input
import const as c

batch_size = 1
SAVE_DIR = '../model_weights'
IMG_PATH = '../9_1_2_face.jpg'
def convert_trt_fp(tfkeras_model_name, trt_type='FP16'):
    assert trt_type in ['FP32', 'FP16', 'INT8'], 'enter correct trt type [FP32, FP16, INT8]'
    trt_model_name = 'TRT'+ trt_type
    if trt_type=='FP16':
        print('Converting to TF-TRT {}...'.format(trt_type))
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            precision_mode=trt.TrtPrecisionMode.FP16,
            max_workspace_size_bytes=8000000000)
        converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=tfkeras_model_name, conversion_params=conversion_params)
        converter.convert()
    elif trt_type=='FP32':
        print('Converting to TF-TRT {}...'.format(trt_type))
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            precision_mode=trt.TrtPrecisionMode.FP32, 
            max_workspace_size_bytes=8000000000)

        converter = trt.TrtGraphConverterV2(input_saved_model_dir=tfkeras_model_name,
                                            conversion_params=conversion_params)
        converter.convert()
    elif trt_type=='INT8':
        img_path = IMG_PATH
        print('Converting to TF-TRT {}...'.format(trt_type))
        batched_input = call_batched_input(img_path=img_path)
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            precision_mode=trt.TrtPrecisionMode.INT8, 
            max_workspace_size_bytes=8000000000, 
            use_calibration=True)
        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=tfkeras_model_name, 
            conversion_params=conversion_params)

        def calibration_input_fn():
            yield (batched_input, )
        converter.convert(calibration_input_fn=calibration_input_fn)
    converter.save(output_saved_model_dir=os.path.join(SAVE_DIR, trt_model_name))
    print('Done Converting to TF-TRT {}'.format(trt_type))
        
if __name__=='__main__':
    args = sys.argv
    tfkeras_model_name = os.path.join(SAVE_DIR, c.TFKERAS_MODEL_NAME)
    trt_type = str(args[1])
    convert_trt_fp(tfkeras_model_name, trt_type=trt_type)

