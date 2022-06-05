import os
import time
import sys
import numpy as np
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.python.saved_model import tag_constants
import const as c

batch_size = 1
def call_batched_input(img_path, size=299):
    batched_input = np.zeros((batch_size, size, size, 3), dtype=np.float32)

    img = image.load_img(img_path, target_size=(size, size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x/255
    batched_input = x.reshape(batch_size, size, size, 3)
    batched_input = tf.constant(batched_input)
    print('batched_input shape: ', batched_input.shape)
    return batched_input 

def trt_model_latency(batched_input, input_saved_model, trt_type):
    saved_model_loaded = tf.saved_model.load(input_saved_model, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']
    labeling = infer(batched_input)

    start_time = time.time()
    labeling = infer(batched_input)
    age_output = labeling['age_output'].numpy()
    gender_output = labeling['gender_output'].numpy()
    gender_output = "M" if np.argmax(gender_output) < 0.5 else "F"
    print('{0} gender {1} and age {2}'.format(trt_type, gender_output, int(age_output*100)))
    predict_time = time.time() - start_time
    print("Inference Latency (milliseconds) is", predict_time*1000, "[ms]")


if __name__=='__main__':
    args = sys.argv
    trt_model_type = str(args[1])
    assert trt_model_type in ['FP32', 'FP16', 'INT8'], 'enter correct type [FP32, FP16, INT8]'
    img_path = '9_1_2_face.jpg'
    trt_type = c.TENSOR_RT_NAME+trt_model_type
    trt_model_dir = 'model_weights/'+c.TENSOR_RT_NAME+trt_model_type
    batched_input = call_batched_input(img_path=img_path)
    trt_model_latency(batched_input, trt_model_dir, trt_type)
