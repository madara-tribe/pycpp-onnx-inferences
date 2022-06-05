import os
import time
import sys
import numpy as np
import cv2

import tensorflow as tf
from tensorflow import keras
import const as c


def load_data(path, size=299):
    x_test = cv2.imread(path)
    x_test = cv2.resize(x_test, (size, size))
    x_test = x_test.astype(np.float32)/255
    return np.expand_dims(x_test, axis=0)
    
# Benchmarking throughput

def tfkeras_latency(model, batched_input):
    start_time = time.time()
    age_output, gender_output = model.predict(batched_input)
    gender_output = "M" if np.argmax(gender_output) < 0.5 else "F"
    print('tfkeras gender {0} and age {1}'.format(gender_output, int(age_output*100)))
    predict_time = time.time() - start_time
    print("Inference Latency (milliseconds) is", predict_time*1000, "[ms]")


if __name__=='__main__':
    output_name = os.path.join('model_weights', c.TFKERAS_MODEL_NAME)
    data_path = '9_1_2_face.jpg'
    print('start tfkeras model latency')
    model = tf.keras.models.load_model(output_name)
    batched_input = load_data(path=data_path)
    tfkeras_latency(model, batched_input)


