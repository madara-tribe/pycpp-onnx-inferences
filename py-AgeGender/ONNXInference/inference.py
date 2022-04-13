import os
os.environ['TF_KERAS'] = '1'
import numpy as np
import cv2
import time
import tensorflow as tf
import onnxruntime
import onnx
from model_utils.load import to_mean_pixel, MEAN_AVG
from model_utils.identity_age import PostProcess, return_generation

ONNX_MODEL_PATH = 'age_gender_model.onnx'
IMAGE_PATH = 'face.jpg'
SIZE = 299
x = 100
y = 250
def draw_label(image, point, label, font=cv2.FONT_HERSHEY_PLAIN,
               font_scale=1.5, thickness=2):
    text_color = (0, 0, 0)
    cv2.putText(image, label, point, font, font_scale, 
            text_color, thickness, lineType=cv2.LINE_AA)


def py_inference():
    img_path = IMAGE_PATH
    onnx_model_path = ONNX_MODEL_PATH
    image = cv2.imread(img_path)
    image = cv2.resize(image, (SIZE, SIZE))
    cimg = image.copy()
    image = to_mean_pixel(image, MEAN_AVG)
    image = image.astype(np.float32)/255
    image = np.expand_dims(image, 0)
    print('onnx input shape', image.shape)
    
    session = onnxruntime.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    generation = session.get_outputs()[0].name
    iden = session.get_outputs()[1].name
    gender_output = session.get_outputs()[2].name
    
    print('start calculation')
    start = time.time() 
    pred_generation, pred_iden, pred_gender = session.run([generation, iden, gender_output], {input_name: image})
    print('len', len(pred_iden[0]), len(pred_iden)) 
    pred_identity = PostProcess(pred_generation, pred_iden).post_age_process()

    pred_gender = "Male" if np.argmax(pred_gender) < 0.5 else "Female"
    print('{0} {1}'.format(pred_gender, pred_identity))
    predict_time = time.time() - start
    print("Inference Latency (milliseconds) is", predict_time*1000, "[ms]")
    
    label = "{0}  {1}".format(int(pred_identity), "Male" if np.argmax(pred_gender) < 0.5 else "Female")

    draw_label(cimg, (x, y), label)
    cv2.imwrite('prediction.png', cimg)

if __name__=='__main__':
    py_inference()
