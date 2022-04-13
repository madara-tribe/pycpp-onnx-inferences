import onnxruntime
import onnx
import cv2
import numpy as np
MEAN_AVG = float(130.509485819935)
def to_mean_pixel(img, avg):
    return (img - 128)*(128/avg)
SIZE = 299

def onnx_inference(path, onnx_model_path):
    image = cv2.imread(path)
    image = cv2.resize(image, (SIZE, SIZE))
    image = to_mean_pixel(image, MEAN_AVG)
    image = image.astype(np.float32)/255
    image = np.expand_dims(image, 0)

    session = onnxruntime.InferenceSession('gender_model.onnx')
    input_name = session.get_inputs()[0].name
    pred_gender = session.run(None, {input_name: image})[0]
    pred_gender = "M" if np.argmax(pred_gender) < 0.5 else "F"
    return pred_gender

if __name__=='__main__':
    path='9_1_3.jpg'
    gender_onnx_path = 'gender_model.onnx'
    pred_gender = onnx_inference(path, gender_onnx_path)
    print(pred_gender)
