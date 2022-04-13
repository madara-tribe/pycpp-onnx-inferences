import sys, cv2
import onnx
import onnxruntime
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import time

transforms_ = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image
    
def load_img(path ='hourse.jpg'):
    transform = transforms.Compose(transforms_)
    img = Image.open(path)
    img = to_rgb(img)
    return transform(img)

def prediction(onnx_file_path):
    inputs = load_img(path = 'hourse.jpg')
    inputs = inputs.to('cpu').detach().numpy().copy()
    inputs = np.expand_dims(inputs, 0)
    print(inputs.shape, inputs.max(), inputs.min())
    model = onnx.load(onnx_file_path)
    onnx.checker.check_model(model)
    print(onnx.helper.printable_graph(model.graph))
    
    print('start calculation')
    start = time.time()
    ort_session = onnxruntime.InferenceSession(onnx_file_path)
    outputs = ort_session.run(None, {"input1": inputs.astype(np.float32)})[0]
    predict_time = time.time() - start
    print("Inference Latency (milliseconds) is", predict_time*1000, "[ms]")

    print(outputs.shape, outputs.min(), outputs.max())
    outputs = outputs * 127.5 + 127.5
    outputs = np.squeeze(outputs.transpose(0, 2, 3, 1), axis=0)
    outputs = cv2.cvtColor(outputs, cv2.COLOR_BGR2RGB)
    cv2.imwrite('2zebra.png', outputs.astype(np.uint8))

if __name__=='__main__':
    onnx_file_path = sys.argv[1]
    prediction(onnx_file_path)
